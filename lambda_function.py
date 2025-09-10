import os
import json
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import boto3

# -------------------------------
# DUMMY KEYS & BUCKET FOR DEMO
# -------------------------------
GEMINI_API_KEY = "DUMMY_GEMINI_API_KEY_123456"
S3_BUCKET = "demo-news-faiss-bucket"
FAISS_FILE_KEY = "faiss_index/index.faiss"

# Initialize S3 client
s3 = boto3.client("s3")

# Example RSS feeds or websites
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.theguardian.com/world/rss"
]

def scrape_rss():
    import feedparser
    docs = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            content = entry.get("summary", "") or entry.get("title", "")
            docs.append(Document(page_content=content, metadata={"source": entry.link}))
    return docs

def scrape_websites():
    urls = ["https://news.ycombinator.com/"]
    docs = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        for item in soup.find_all("a", class_="storylink"):
            docs.append(Document(page_content=item.text, metadata={"source": item['href']}))
    return docs

def lambda_handler(event, context):
    try:
        # Step 1: Scrape articles
        rss_docs = scrape_rss()
        web_docs = scrape_websites()
        all_docs = rss_docs + web_docs
        if not all_docs:
            return {"statusCode": 200, "body": "No articles found."}

        # Step 2: Split articles into chunks
        splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        chunks = splitter.split_documents(all_docs)

        # Step 3: Generate embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Step 4: Save FAISS index to temporary file and upload to S3
        with tempfile.TemporaryDirectory() as tmpdirname:
            faiss_path = f"{tmpdirname}/index.faiss"
            vectorstore.save_local(faiss_path)
            s3.upload_file(faiss_path, S3_BUCKET, FAISS_FILE_KEY)

        return {
            "statusCode": 200,
            "body": f"✅ FAISS index updated with {len(chunks)} chunks and uploaded to S3 ({S3_BUCKET})."
        }

    except Exception as e:
        return {"statusCode": 500, "body": f"⚠️ Error: {str(e)}"}
