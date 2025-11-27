from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from llama_index.llms.google_genai import GoogleGenAI
import llama_index.llms.google_genai
# 1. 配置全局嵌入模型
Settings.llm = GoogleGenAI(
    model="models/gemini-2.5-flash",  
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    max_tokens=1024,
)
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于描述 lama的 large指数， index越大，说明获得了越大的lama",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

# 4. 正确加载本地索引并执行查询
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
loaded_index = load_index_from_storage(storage_context)

# 5. 创建查询引擎,设置返回最相关的2个文本块
query_engine = loaded_index.as_query_engine(
    similarity_top_k=2  # 检索最相关的2个文本块
)

# 6. 执行查询
query = "LlamaIndex是什么？"
print(f"\n查询问题: '{query}'")
print("=" * 60)

response = query_engine.query(query)

# 7. 打印查询结果
print("\nLLM 回答:")
print(response)
print("\n" + "=" * 60)

# 8. 查看检索到的源文档(最相关的2个文本块)
print("\n检索到的相关文本块:")
for i, node in enumerate(response.source_nodes, 1):
    print(f"\n文本块 {i} (相似度分数: {node.score:.4f}):")
    print(f"  内容: {node.text}")
