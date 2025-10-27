# ingest.py
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define the path for the persistent vector database
PERSIST_DIRECTORY = 'db'
DATA_PATH = 'data'

def main():
    print("Starting ingestion process...")

    # 1. Load Documents
    print(f"Loading documents from {DATA_PATH}...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",  # Look for PDF files
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    if not documents:
        print("No PDF documents found. Exiting.")
        return
    print(f"Loaded {len(documents)} documents.")

    # 2. Split Documents into Chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # 3. Create Embeddings
    print("Creating embeddings... (This may take a while for the first time)")
    # We will use a local model, so no API keys are needed
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embeddings model loaded.")

    # 4. Store Chunks in a Vector Database (ChromaDB)
    print(f"Storing embeddings in vector store at {PERSIST_DIRECTORY}...")
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    print("Ingestion complete!")
    print(f"Vector store created at '{PERSIST_DIRECTORY}' with {db._collection.count()} vectors.")

if __name__ == "__main__":
    main()