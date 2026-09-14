import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="RAG 知识库问答系统")

embed_client = OpenAI(
      api_key=os.getenv("SILICONFLOW_API_KEY"),
      base_url="https://api.siliconflow.cn/v1"
  )
chat_client = OpenAI(
      api_key=os.getenv("DEEPSEEK_API_KEY"),
      base_url="https://api.deepseek.com"
  )

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_docs")


def get_embedding(text):
      resp = embed_client.embeddings.create(model="BAAI/bge-m3", input=text)
      return resp.data[0].embedding


def split_text(text, chunk_size=200, overlap=50):
      chunks = []
      start = 0
      while start < len(text):
          end = start + chunk_size
          chunks.append(text[start:end])
          if end >= len(text):
              break
          start = end - overlap
      return chunks


def ingest(file_path="my_doc.txt"):
      with open(file_path, "r", encoding="utf-8") as f:
          text = f.read()
      chunks = split_text(text)
      embeddings = [get_embedding(c) for c in chunks]
      collection.add(
          documents=chunks,
          embeddings=embeddings,
          ids=[f"chunk_{i}" for i in range(len(chunks))]
      )
      return len(chunks)


  # 启动时如果库是空的，自动入库
if collection.count() == 0:
      n = ingest()
      print(f"已入库 {n} 块文档")


def answer_question(question):
      q_embedding = get_embedding(question)
      results = collection.query(query_embeddings=[q_embedding], n_results=3)
      docs = results["documents"][0]
      material = "\n\n".join(docs)
      prompt = f"""你是一个知识库问答助手。请只根据下面提供的资料回答问题。
  如果资料中没有相关信息，请直接回答"资料中没有相关内容"，不要编造。

  【资料】
  {material}

  【问题】
  {question}
  """
      resp = chat_client.chat.completions.create(
          model="deepseek-chat",
          temperature=0,
          messages=[{"role": "user", "content": prompt}]
      )
      return resp.choices[0].message.content


class Question(BaseModel):
      question: str


@app.post("/ask")
def ask(q: Question):
      return {"answer": answer_question(q.question)}


@app.get("/")
def home():
      return {"message": "RAG 知识库问答系统已运行，访问 /docs 测试"}