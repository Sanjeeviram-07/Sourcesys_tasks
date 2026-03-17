# import streamlit as st
# import os
# from dotenv import load_dotenv

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA

# # Load API key
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

# st.title("📚 AI Document Assistant (Free RAG Project)")

# # -----------------------------
# # 1️⃣ Load Document
# # -----------------------------

# loader = TextLoader("data.txt")
# documents = loader.load()

# # -----------------------------
# # 2️⃣ Split into chunks
# # -----------------------------

# text_splitter = CharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )

# docs = text_splitter.split_documents(documents)

# # -----------------------------
# # 3️⃣ Create Embeddings
# # -----------------------------

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # -----------------------------
# # 4️⃣ Create Vector Database
# # -----------------------------

# vector_db = FAISS.from_documents(docs, embedding_model)

# retriever = vector_db.as_retriever()

# # -----------------------------
# # 5️⃣ Load LLM (Gemini)
# # -----------------------------

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     google_api_key=api_key
# )

# # -----------------------------
# # 6️⃣ Build RAG Chain
# # -----------------------------

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever
# )

# # -----------------------------
# # 7️⃣ User Interface
# # -----------------------------

# question = st.text_input("Ask a question about the document:")

# if question:
#     answer = qa_chain.run(question)
#     st.write("### 🤖 Answer:")
#     st.write(answer)
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.title("📚 AI Document Assistant (RAG with Groq)")

# -----------------------------
# Load Document
# -----------------------------

loader = TextLoader("data.txt")
documents = loader.load()

# -----------------------------
# Split Text into Chunks
# -----------------------------

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# -----------------------------
# Create Embeddings
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create Vector Database
# -----------------------------

vector_db = FAISS.from_documents(docs, embedding_model)

retriever = vector_db.as_retriever()

# -----------------------------
# Load Groq LLM
# -----------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# -----------------------------
# Build RAG Chain
# -----------------------------

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# -----------------------------
# Streamlit UI
# -----------------------------

question = st.text_input("Ask a question about the document")

if question:
    answer = qa_chain.run(question)

    st.write("### 🤖 Answer")
    st.write(answer)