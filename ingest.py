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
from langchain_community.vectorstores import Chroma

DOCUMENTS_PATH = "documents"
CRHOMA_PATH = "chroma_db"

# returns correct loader based off file type
# else, returns None if file type is not supported
def get_loader(file_path):
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