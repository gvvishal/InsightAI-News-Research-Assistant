import feedparser
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FAISS_PATH = "faiss_index"

# Fetch articles from RSS
def fetch_articles_from_rss():
    from rss_sources import RSS_FEEDS
    docs = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            content = entry.get("summary", "") or entry.get("title", "")
            docs.append(Document(page_content=content, metadata={"source": entry.link}))
    return docs

# Split articles
def split_articles(docs):
    splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    return splitter.split_documents(docs)

# Update FAISS index
def update_faiss_index():
    docs = fetch_articles_from_rss()
    if not docs:
        print("No articles found!")
        return
    chunks = split_articles(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_PATH)
    print(f"✅ FAISS index updated with {len(chunks)} chunks")
