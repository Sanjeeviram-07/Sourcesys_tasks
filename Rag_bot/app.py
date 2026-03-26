import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


st.title("📚 Document RAG Assistant")
st.write("Upload a PDF or DOCX and ask questions from it.")


# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx"]
)

if uploaded_file:

    # Save uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # -----------------------------
    # Load Document
    # -----------------------------
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file.name)

    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(uploaded_file.name)

    documents = loader.load()

    # -----------------------------
    # Split Text
    # -----------------------------
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # -----------------------------
    # Embeddings
    # -----------------------------
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Vector DB
    # -----------------------------
    vector_db = FAISS.from_documents(docs, embedding_model)

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    # -----------------------------
    # Groq LLM
    # -----------------------------
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    # -----------------------------
    # Prompt
    # -----------------------------
    template = """
You are an AI assistant answering questions from a document.

Use ONLY the provided context.

If the answer is not present in the document, reply:
"I don't know based on the document."

Context:
{context}

Question:
{input}

Answer:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "input"]
    )

    # -----------------------------
    # RAG Chain
    # -----------------------------
    document_chain = create_stuff_documents_chain(llm, prompt)

    qa_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # -----------------------------
    # Question Input
    # -----------------------------
    question = st.text_input("Ask a question from the document")

    if question:

        response = qa_chain.invoke({"input": question})

        answer = response["answer"]

        st.write("### Answer")
        st.write(answer)
