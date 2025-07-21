import os
import hashlib
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Get all pdf paths
def get_all_pdfs_in_folder(folder_path: str) -> list[str]:
    folder = Path(folder_path)
    pdf_files = list(map(str, folder.glob("*.pdf")))
    return pdf_files

# One-time setup to build and persist vector index
def build_vectorstore(pdf_folder: str, index_path: str = "vector_db"):

    # All pdf paths
    pdf_paths = get_all_pdfs_in_folder(pdf_folder)

    # Load all documents
    all_docs = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)

# Load existing vectorstore
def load_vectorstore(index_path: str = "vector_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Search context based on a query
def retrieve_context(query: str, vectorstore, k: int = 3) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)
