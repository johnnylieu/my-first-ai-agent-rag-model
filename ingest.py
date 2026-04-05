import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCUMENTS_PATH = "documents"
CHROMA_PATH = "chroma_db"

def get_loader(file_path):
    """
    returns correct loader based off file type
    else, returns None if file type is not supported
    """

    extension = os.path.splitext(file_path)[1].lower()

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
        ".html": UnstructuredHTMLLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pptx": UnstructuredPowerPointLoader
    }

    loader_class = loaders.get(extension)

    if loader_class is None:
        print(f"Unsupported file type: {extension}")
        return None
    
    return loader_class(file_path)

def ingest_documents():
    """
    chunk overlap is used to preserve context across chunk boundaries
    without it, a sentence that spans two chunks could lose its meaning when retreived in isolation
    """
    documents = []

    for filename in os.listdir(DOCUMENTS_PATH):
        file_path = os.path.join(DOCUMENTS_PATH, filename)
        loader = get_loader(file_path)

        if loader is None:
            continue

        loaded_docs = loader.load()
        documents.extend(loaded_docs)
        print(f"Loaded: {filename}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 150,
        chunk_overlap=20
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    return chunks

def store_in_chroma(chunks):
    """
    uses the embedding moel to convert document chunks into vectors
    stores the resulting vectorstore in CHROMA_PATH using ChromaDB
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

if __name__ == "__main__":
    chunks = ingest_documents()
    store_in_chroma(chunks)
    print("Ingestion complete!")