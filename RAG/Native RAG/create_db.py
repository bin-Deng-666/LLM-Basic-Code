from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量

CHROMA_PATH = "./database"

if __name__ == "__main__":
    # Make a new directory
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Load and split documents
    # 检查文件是否存在
    if not os.path.exists("data/KB.txt"):
        print("错误: data/KB.txt 文件不存在")
        print("请确保在data目录中创建了KB.txt文件并添加了您的知识库内容")
        exit(1)
    
    loader = TextLoader("data/KB.txt")
    documents = loader.load()
    
    # 检查是否成功加载了文档
    if not documents or len(documents) == 0:
        print("警告: 未能从文件中加载任何内容")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("请在.env文件中设置或直接在代码中提供API密钥")
    
    embedding_function = OpenAIEmbeddings(
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3-Embedding-0.6B",
        api_key=api_key  # 使用环境变量中的API密钥
    )
    
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    
    db.persist()
