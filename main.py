import os
import streamlit as st
import time
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🚨 API Key not found! Please check your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Set page title and layout
st.set_page_config(page_title="InsightAI: News Research Assistant", page_icon="📊", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        .stTextInput>div>div>input { font-size: 18px; padding: 10px; }
        .stButton>button { font-size: 16px; border-radius: 10px; padding: 8px 20px; background-color: #4CAF50; color: white; border: none; }
        .stButton>button:hover { background-color: #45a049; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Title
st.title(" InsightAI: News Research Assistant")
st.markdown("### Analyze and extract key insights from news articles in real-time!")

# Sidebar for URL input
st.sidebar.header("🔗 Input News Article URLs")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button(" Process URLs")
faiss_store_path = "faiss_store_gemini"

# Status Placeholder
status_placeholder = st.empty()

if process_url_clicked:
    if not any(urls):
        st.error("🚨 Please enter at least one valid URL before processing.")
    else:
        status_placeholder.info("🔄 Loading data...")
        try:
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            
            status_placeholder.info("🔄 Splitting text...")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)
            
            status_placeholder.info("🔄 Generating embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=GEMINI_API_KEY
            )
            vectorstore_gemini = FAISS.from_documents(docs, embeddings)
            
            # Save FAISS index safely
            vectorstore_gemini.save_local(faiss_store_path)
            
            status_placeholder.success("✅ Data processed successfully!")
            time.sleep(1)
        except Exception as e:
            st.error(f"⚠️ Error processing data: {str(e)}")

# Question Input
query = st.text_input("💬 Ask a Question about the News Articles:")

if query:
    if os.path.exists(faiss_store_path):
        try:
            # Load FAISS index safely
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=GEMINI_API_KEY
            )
            vectorstore = FAISS.load_local(
                faiss_store_path, embeddings, allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever()

            # Retrieve relevant documents
            relevant_docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            if not context:
                st.error("🚨 No relevant data found. Try processing URLs again.")
            else:
                # Generate response with context
                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content(f"Context:\n{context}\n\nQuestion: {query}")

                st.markdown("## 📌 Answer")
                st.write(response.text)

        except Exception as e:
            st.error(f"⚠️ Error generating response: {str(e)}")
    else:
        st.error("🚨 No processed data found! Please process URLs first.")
