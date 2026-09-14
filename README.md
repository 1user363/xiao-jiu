# RAG 知识库问答系统

基于 **检索增强生成（RAG）** 的私有知识库问答系统。将文档切分、向量化存入向量数据库，用户提问时检索最相关内容，交给大模型基于资料生成答案，解决大模型知识过时、幻觉等问题。

## 功能特性

- 文档自动切分与向量化入库（启动时自动完成）
- 基于语义相似度的向量检索（ChromaDB + BAAI/bge-m3）
- 基于检索内容的生成式问答（DeepSeek）
- 防幻觉：资料中没有的内容会明确回答「没有」，不会编造
- FastAPI 提供 REST 接口，自动生成 `/docs` 接口文档

## 技术栈

- **语言 / 框架**：Python、FastAPI、Uvicorn
- **大模型**：DeepSeek（对话）+ 硅基流动 BAAI/bge-m3（Embedding）
- **向量数据库**：ChromaDB
- **SDK**：OpenAI SDK（兼容接口）

## 工作流程

```
              【离线准备（启动时自动）】
文档(my_doc.txt) → 切分(chunk) → 向量化(Embedding) → 存入 ChromaDB

              【在线问答（每次请求）】
提问 → 向量化 → 相似度检索 Top3 → 拼接 Prompt → DeepSeek 生成答案
```

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置密钥：新建 `.env` 文件，填入（`DEEPSEEK_API_KEY` 从 platform.deepseek.com 获取，`SILICONFLOW_API_KEY` 从 siliconflow.cn 获取）：

```
DEEPSEEK_API_KEY=sk-xxx
SILICONFLOW_API_KEY=sk-xxx
```

3. 放入文档：把要问答的文档命名为 `my_doc.txt` 放在项目根目录

4. 启动服务

```bash
python -m uvicorn app:app --reload
```

5. 打开 http://127.0.0.1:8000/docs 测试接口

## API

### POST /ask

请求：

```json
{"question": "什么是RAG？"}
```

响应：

```json
{"answer": "RAG 是检索增强生成技术，..."}
```

## 项目说明

- 向量库存在本地 `chroma_db/`，首次启动时若库为空会自动入库
- `temperature=0` 保证回答稳定，Prompt 中强制「资料没有就说没有」，避免幻觉
