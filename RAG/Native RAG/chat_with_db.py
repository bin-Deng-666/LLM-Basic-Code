from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

# 向量存储库路径
CHROMA_PATH = "database"
# API Key
API_KEY = os.getenv("SILICONFLOW_API_KEY")

# 定义对话提示词模版
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

------

Answer the question based on the above context: {question}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    # 获得问题
    query_text = args.query_text

    # 验证API密钥
    if not API_KEY:
        print("错误: 请在.env文件中设置SILICONFLOW_API_KEY")
        exit(1)
    
    # 准备给予Chroma的向量检索库
    embeddings = OpenAIEmbeddings(
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3-Embedding-0.6B",
        api_key=API_KEY  # 使用API密钥
    )
    db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)

    # 检索向量库
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # 检索出来的内容
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    
    # 生成回答
    llm = ChatOpenAI(
        base_url="https://api.siliconflow.cn/v1",
        api_key=API_KEY,
        model="deepseek-ai/DeepSeek-V3",
        temperature=0.7,
        max_tokens=2048
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    print(response.content)