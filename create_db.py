from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

CHROMA_PATH = "database"
DATA_PATH = "data/books"
MODEL_PATH = "YOUR_MODEL_PATH"

if __name__ == "__main__":
    # Load and split documents
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embedding_function = SentenceTransformerEmbeddings(model_name=MODEL_PATH)
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    
    # Persist the database
    db.persist()
