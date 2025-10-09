from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os
from dotenv import load_dotenv
load_dotenv()

# 向量存储库路径
CHROMA_PATH = "./database"
# API Key
API_KEY = os.getenv("SILICONFLOW_API_KEY")

if __name__ == "__main__":
    # 删除向量检索库 
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # 加载文件
    loader = TextLoader("data/KB.txt")
    documents = loader.load()

    # 文件切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # 定义Embedding模型并进行文件片向量化
    embedding_function = OpenAIEmbeddings(
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3-Embedding-0.6B",
        api_key=API_KEY
    )
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    
    # 存储向量库
    db.persist()
