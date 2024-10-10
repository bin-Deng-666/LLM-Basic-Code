from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList
from typing import Optional, Union
import argparse

EMB_MODEL_PATH = "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/dengbin16/langchain-RAG/model/all-MiniLM-L6-v2"
CHAT_MODEL_PATH = "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/fengyizhe2001/GLM-4-main/glm-4-9b-chat"
CHROMA_PATH = "database"
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
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = SentenceTransformerEmbeddings(model_name=EMB_MODEL_PATH)
    db = Chroma(embedding_function=embedding_function, persist_directory=CHROMA_PATH)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # Make the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_PATH, trust_remote_code=True, encode_special_tokens=True)
    model = AutoModel.from_pretrained(CHAT_MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
    
    message = [
        {"role": "user", "content": prompt}
    ]
    
    input_message = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    max_input_tokens = len(input_message)
    
    gen_kwargs = {
        "max_new_tokens": 8192,
        "num_beams": 1,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "eos_token_id": model.config.eos_token_id
    }
    
    inputs = tokenizer(input_message, return_tensors="pt", padding="max_length", truncation=True, max_length=max_input_tokens).to(model.device)
    outputs = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(response)
    
