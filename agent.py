from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
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

def create_prompt():
    template = """
    You are helpful assistant, you will be answering questions about Johnny Lieu. 
    Use the following context to answer the question at the end. If you don't 
    know the answer based on the context, just say you don't know.
    Don't make anything up.

    context: {context}

    Question: {question}

    Answer:
    """

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def create_chain():
    """
    Wires load_retriever and create_prompt together into a pipeline
    """
    retriever = load_retriever()
    prompt = create_prompt()
    llm = ChatOllama(model="llama3")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__ == "__main__":
    chain = create_chain()

    print("Agent ready. Type 'quit' to exit.")

    while True:
        question = input("\nWhat's on your mind?: ")

        if question.lower() == "quit":
            break

        response = chain.invoke(question)
        print(f"\nAgent's response: {response}")