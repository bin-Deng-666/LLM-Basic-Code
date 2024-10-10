from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import shutil
import os


CHROMA_PATH = "database"
DATA_PATH = "data"
EMB_MODEL_PATH = "YOUR_TEXT_EMBEDDING_MODEL_PATHR_"

if __name__ == "__main__":
    # Make a new directory
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Load and split documents
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embedding_function = SentenceTransformerEmbeddings(model_name=EMB_MODEL_PATH)
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    
    # Persist the database
    db.persist()
