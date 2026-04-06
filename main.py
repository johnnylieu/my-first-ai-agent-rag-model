from ingest import ingest_documents, store_in_chroma
from agent import create_chain
import os

def main():
    # step 1 - check if ChromaDB already exists or if folder is empty
    if not os.path.exists("chroma_db") or not os.listdir("chroma_db"):
        print("No knowledge base found. Ingesting documents, nom nom nom...")
        chunks = ingest_documents()
        store_in_chroma(chunks)
        print("Deliscious. Knowledge base created.\n")
    else:
        print("Knowledge base found. Skipping ingestion.\n")

    # step 2 - start the agent
    print("Starting agent...")
    chain = create_chain()
    print("Agent is ready. Type 'quit' to exit.\n")

    # step 3 - chat loop
    while True:
        question = input("What's on your mind?: ")

        if question.lower() == "quit":
            print("Good bye! Please keep in mind that I'm still learning. I recommend reading Mr. Lieu's resume.")
            break

        response = chain.invoke(question)
        print(f"\nAgent's response: {response} \n")


if __name__ == "__main__":
    main()