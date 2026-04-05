import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "chroma_db"

def load_retriever():
    """
    connects to our ChromaDB vectorstore and returns a retriever
    configured to fetch the 5 most chunks for any given question
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})