# 4.3 RAG 应用开发

> 检索增强生成（RAG）结合了检索系统的准确性和大模型的生成能力，是企业应用的主流架构。

## 学习目标

学完本节后，你将能够：

- [ ] 理解 RAG 架构的核心组件
- [ ] 选择合适的向量化模型和数据库
- [ ] 实施混合检索策略
- [ ] 优化 RAG 系统的召回和生成质量

---

## 4.3.1 RAG 架构概述

```
                    RAG 系统架构

┌─────────────────────────────────────────────────────────┐
│                      用户查询                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  1. 查询处理                                             │
│     - 查询重写/扩展                                       │
│     - 意图识别                                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  2. 检索                                                 │
│     ┌──────────────┐    ┌──────────────┐               │
│     │  向量检索     │    │  关键词检索   │               │
│     │  (语义匹配)   │    │  (BM25)      │               │
│     └──────┬───────┘    └───────┬──────┘               │
│            └──────────┬──────────┘                      │
│                       │                                  │
│                       ▼                                  │
│              ┌────────────────┐                         │
│              │  结果融合/重排  │                         │
│              └────────┬───────┘                         │
└───────────────────────┼─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  3. 生成                                                 │
│     ┌──────────────────────────────────────┐            │
│     │  Prompt:                             │            │
│     │  基于以下文档回答问题：               │            │
│     │  [相关文档 1]                        │            │
│     │  [相关文档 2]                        │            │
│     │  问题：{query}                       │            │
│     └──────────────────────────────────────┘            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
              最终答案
```

---

## 4.3.2 向量化与 Embedding

### 主流 Embedding 模型对比

| 模型 | 维度 | 最大长度 | MTEB 分数 | 适用场景 |
|------|------|----------|-----------|----------|
| **text-embedding-3-large** | 3072 | 8191 | 64.6 | 通用 |
| **BGE-large-zh-v1.5** | 1024 | 512 | 64.3 | 中文 |
| **M3E-base** | 768 | 512 | 63.1 | 中文 |
| **Jina-embeddings-v2** | 1024 | 8192 | 62.8 | 长文本 |

### 向量化实现

```python
# examples/4-3-rag/embedding.py
"""
向量化与 Embedding
"""

import numpy as np
from typing import List, Union
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    model_name: str = "BAAI/bge-large-zh-v1.5"
    dimension: int = 1024
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda"


class EmbeddingModel:
    """
    Embedding 模型封装
    
    支持：
    - Sentence Transformers
    - OpenAI Embedding API
    - 自定义模型
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        if config.model_name.startswith("openai/"):
            self._init_openai(config)
        else:
            self._init_sentence_transformer(config)
    
    def _init_openai(self, config: EmbeddingConfig):
        """初始化 OpenAI Embedding"""
        from openai import OpenAI
        
        self.client = OpenAI()
        self.model = config.model_name.replace("openai/", "")
    
    def _init_sentence_transformer(self, config: EmbeddingConfig):
        """初始化 Sentence Transformer"""
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(
            config.model_name,
            device=config.device
        )
    
    def encode(self, 
               texts: Union[str, List[str]],
               normalize: bool = True,
               batch_size: int = None) -> np.ndarray:
        """
        向量化文本
        
        参数:
            texts: 单个文本或文本列表
            normalize: 是否 L2 归一化（用于余弦相似度）
            batch_size: 批处理大小
        """
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = batch_size or self.config.batch_size
        
        if hasattr(self, 'client'):
            # OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            embeddings = np.array([
                item.embedding for item in response.data
            ])
        else:
            # 本地模型
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > batch_size
            )
        
        return embeddings
    
    def encode_query(self, text: str) -> np.ndarray:
        """编码查询（某些模型需要特殊处理）"""
        # BGE 等模型需要添加前缀
        if "bge" in self.config.model_name.lower():
            text = "为这个句子生成表示以用于检索：" + text
        
        return self.encode(text)[0]
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """编码文档"""
        # BGE 等模型需要添加前缀
        if "bge" in self.config.model_name.lower():
            documents = ["代表这个句子：" + d for d in documents]
        
        return self.encode(documents)
    
    def similarity(self, 
                   query: np.ndarray, 
                   documents: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度
        
        返回:
            每个文档与查询的相似度分数
        """
        # 确保归一化
        query_norm = query / np.linalg.norm(query)
        docs_norm = documents / np.linalg.norm(documents, axis=1, keepdims=True)
        
        return docs_norm @ query_norm


# 使用示例
async def demo_embedding():
    """Embedding 使用演示"""
    
    config = EmbeddingConfig(
        model_name="BAAI/bge-large-zh-v1.5",
        dimension=1024
    )
    
    embedder = EmbeddingModel(config)
    
    # 编码文档
    documents = [
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的核心技术。",
        "深度学习是机器学习的子领域。"
    ]
    
    doc_embeddings = embedder.encode_documents(documents)
    print(f"文档向量形状：{doc_embeddings.shape}")
    
    # 编码查询
    query = "什么是人工智能？"
    query_embedding = embedder.encode_query(query)
    print(f"查询向量形状：{query_embedding.shape}")
    
    # 计算相似度
    similarities = embedder.similarity(query_embedding, doc_embeddings)
    print(f"相似度分数：{similarities}")
    
    # 最相似的文档
    top_idx = np.argmax(similarities)
    print(f"最相关文档：{documents[top_idx]}")
```

---

## 4.3.3 向量数据库

### 主流向量数据库对比

| 数据库 | 开源 | 索引类型 | 规模 | 特色 |
|--------|------|----------|------|------|
| **Chroma** | ✅ | HNSW | 百万级 | 轻量、易上手 |
| **FAISS** | ✅ | IVF/HNSW | 十亿级 | Facebook、高性能 |
| **Milvus** | ✅ | 多种 | 十亿级 | 功能完整 |
| **Qdrant** | ✅ | HNSW | 千万级 | Rust、RESTful |
| **Pinecone** | ❌ | 自研 | 十亿级 | 托管服务 |

### Chroma 实现

```python
# examples/4-3-rag/vector_store.py
"""
使用 Chroma 向量数据库
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict
import uuid


class VectorStore:
    """
    向量存储与检索
    
    功能:
    - 文档存储
    - 相似度检索
    - 元数据过滤
    """
    
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 embedding_fn=None):
        """
        参数:
            collection_name: 集合名称
            persist_directory: 持久化路径
            embedding_fn: 嵌入函数
        """
        
        # 客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        # 集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 余弦相似度
        )
        
        self.embedding_fn = embedding_fn
    
    def add_documents(self,
                      documents: List[str],
                      metadatas: List[Dict] = None,
                      ids: List[str] = None,
                      batch_size: int = 100):
        """
        添加文档
        
        参数:
            documents: 文档内容
            metadatas: 元数据列表
            ids: 文档 ID（不传则自动生成）
        """
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # 批量添加
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size] if metadatas else None
            
            # 向量化
            if self.embedding_fn:
                embeddings = self.embedding_fn.encode(batch_docs)
            else:
                embeddings = None
            
            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids
            )
        
        print(f"已添加 {len(documents)} 个文档")
    
    def search(self,
               query: str,
               n_results: int = 5,
               where: Dict = None,
               include: List[str] = None) -> Dict:
        """
        相似度搜索
        
        参数:
            query: 查询文本
            n_results: 返回数量
            where: 元数据过滤条件
            include: 返回字段
        
        返回:
            {
                'documents': [...],
                'metadatas': [...],
                'distances': [...],
                'ids': [...]
            }
        """
        
        include = include or ["documents", "metadatas", "distances"]
        
        # 向量化查询
        if self.embedding_fn:
            query_embedding = self.embedding_fn.encode([query])
        else:
            query_embedding = None
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=include
        )
        
        return results
    
    def hybrid_search(self,
                      query: str,
                      bm25_index,
                      n_results: int = 5,
                      alpha: float = 0.5) -> List[Dict]:
        """
        混合检索：向量 + BM25
        
        参数:
            query: 查询文本
            bm25_index: BM25 索引
            n_results: 返回数量
            alpha: 融合权重（0=纯 BM25, 1=纯向量）
        
        返回:
            融合后的结果列表
        """
        
        # 向量检索
        vector_results = self.search(query, n_results=n_results * 2)
        
        # BM25 检索
        bm25_results = bm25_index.search(query, k=n_results * 2)
        
        # 融合（RRF - Reciprocal Rank Fusion）
        fused_scores = {}
        
        for i, doc_id in enumerate(vector_results['ids'][0]):
            rank = i + 1
            score = 1.0 / (rank + 60)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + alpha * score
        
        for i, doc_id in enumerate(bm25_results['ids']):
            rank = i + 1
            score = 1.0 / (rank + 60)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1 - alpha) * score
        
        # 排序取 Top-K
        sorted_docs = sorted(fused_scores.items(), key=lambda x: -x[1])[:n_results]
        
        return sorted_docs
    
    def delete_collection(self):
        """删除集合"""
        self.client.delete_collection(self.collection.name)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "count": self.collection.count(),
            "name": self.collection.name,
        }


# BM25 索引实现
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 关键词索引"""
    
    def __init__(self, documents: List[str]):
        """
        参数:
            documents: 文档列表
        """
        # 分词
        self.tokenized_docs = [self._tokenize(d) for d in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.documents = documents
    
    def _tokenize(self, text: str) -> List[str]:
        """分词（中文需要分词器）"""
        import jieba
        return list(jieba.cut(text))
    
    def search(self, query: str, k: int = 5) -> Dict:
        """
        BM25 搜索
        
        返回:
            {
                'ids': [...],
                'scores': [...],
                'documents': [...]
            }
        """
        query_tokens = self._tokenize(query)
        
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:k]
        
        return {
            'ids': list(top_indices),
            'scores': scores[top_indices],
            'documents': [self.documents[i] for i in top_indices]
        }
```

---

## 4.3.4 完整 RAG 系统

```python
# examples/4-3-rag/rag_system.py
"""
完整 RAG 系统实现
"""

from dataclasses import dataclass
from typing import List, Optional
import asyncio


@dataclass
class RAGConfig:
    """RAG 配置"""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    llm_model: str = "gpt-4o"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    rerank: bool = True
    rerank_top_k: int = 3


class RAGSystem:
    """
    检索增强生成系统
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # 初始化组件
        self.embedder = EmbeddingModel(
            EmbeddingConfig(model_name=config.embedding_model)
        )
        
        self.vector_store = VectorStore(
            embedding_fn=self.embedder
        )
        
        self.llm_client = OpenAIClient(api_key="xxx")
        
        # 可选的重排序模型
        if config.rerank:
            self.reranker = self._init_reranker()
        else:
            self.reranker = None
    
    def _init_reranker(self):
        """初始化重排序模型"""
        from sentence_transformers import CrossEncoder
        
        return CrossEncoder('BAAI/bge-reranker-large')
    
    def ingest_documents(self, documents: List[str]):
        """
        导入文档
        
        流程:
        1. 分块
        2. 向量化
        3. 存储
        """
        # 分块
        chunks = self._chunk_documents(documents)
        
        # 存储
        self.vector_store.add_documents(
            documents=chunks,
            metadatas=[{"source": i} for i in range(len(chunks))]
        )
        
        print(f"已导入 {len(chunks)} 个文档块")
    
    def _chunk_documents(self, 
                         documents: List[str],
                         chunk_size: int = None,
                         overlap: int = None) -> List[str]:
        """
        文档分块
        
        Overlap 确保上下文连续
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        
        chunks = []
        
        for doc in documents:
            # 简单按字符分块（实际应用应基于语义）
            start = 0
            while start < len(doc):
                end = start + chunk_size
                chunk = doc[start:end]
                
                if chunk.strip():
                    chunks.append(chunk.strip())
                
                start += chunk_size - overlap
        
        return chunks
    
    async def query(self, 
                    question: str,
                    n_results: int = None) -> str:
        """
        查询问答
        
        流程:
        1. 检索相关文档
        2. (可选) 重排序
        3. 构建 Prompt
        4. LLM 生成答案
        """
        n_results = n_results or self.config.top_k
        
        # 1. 检索
        results = self.vector_store.search(
            query=question,
            n_results=n_results * 2 if self.reranker else n_results
        )
        
        # 2. 重排序（可选）
        if self.reranker and len(results['documents'][0]) > 0:
            documents = results['documents'][0]
            
            # 计算重排序分数
            pairs = [[question, doc] for doc in documents]
            scores = self.reranker.predict(pairs)
            
            # 取 Top-K
            top_indices = scores.argsort()[::-1][:self.config.rerank_top_k]
            contexts = [documents[i] for i in top_indices]
        else:
            contexts = results['documents'][0][:n_results]
        
        # 3. 构建 Prompt
        prompt = self._build_rag_prompt(question, contexts)
        
        # 4. 生成答案
        answer = await self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}]
        )
        
        return answer
    
    def _build_rag_prompt(self, 
                          question: str, 
                          contexts: List[str]) -> str:
        """
        构建 RAG Prompt
        """
        context_text = "\n\n".join(
            f"[文档 {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        )
        
        prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请说明。

<documents>
{context_text}
</documents>

问题：{question}

请用中文回答，并引用相关文档的来源。"""

        return prompt
    
    def stream_query(self, question: str):
        """流式查询"""
        return self.llm_client.chat_stream(
            messages=[{
                "role": "user", 
                "content": self._build_rag_prompt(
                    question, 
                    ["检索中..."]
                )
            }]
        )


# 使用示例
async def demo_rag():
    """RAG 系统演示"""
    
    config = RAGConfig(
        embedding_model="BAAI/bge-large-zh-v1.5",
        top_k=5,
        rerank=True
    )
    
    rag = RAGSystem(config)
    
    # 导入文档
    documents = [
        "人工智能（AI）是计算机科学的一个分支，研究如何使计算机模拟人类智能。",
        "机器学习是 AI 的核心，通过数据训练模型来实现预测和决策。",
        "深度学习是机器学习的子领域，使用多层神经网络处理复杂任务。"
    ]
    
    rag.ingest_documents(documents)
    
    # 查询
    question = "什么是人工智能？"
    answer = await rag.query(question)
    
    print(f"问题：{question}")
    print(f"答案：{answer}")
```

---

## 4.3.5 RAG 优化技巧

### 1. 查询重写

```python
# 查询扩展
def expand_query(query: str, n_variations: int = 3) -> List[str]:
    """生成查询变体"""
    
    llm = OpenAIClient()
    
    prompt = f"""为以下问题生成 {n_variations} 个不同表述的查询：
原查询：{query}

要求：
- 保持语义不变
- 使用不同的词汇和句式
- 每个变体 10-20 字

变体:"""
    
    response = await llm.chat([{"role": "user", "content": prompt}])
    
    return response.split("\n")
```

### 2. 元数据过滤

```python
# 添加时间范围过滤
results = vector_store.search(
    query="AI 技术",
    where={
        "date": {"$gte": "2024-01-01"},
        "category": "technology"
    }
)
```

### 3. 评估 RAG 质量

```python
def evaluate_rag(retrieved_docs: List[str], 
                 ground_truth: str) -> dict:
    """评估 RAG 质量"""
    
    from ragas import EvaluationDataset
    from ragas.metrics import context_precision, answer_relevancy
    
    # 使用 Ragas 评估
    metrics = {
        "context_precision": context_precision,
        "answer_relevancy": answer_relevancy,
    }
    
    # 计算分数
    # ...
    
    return scores
```

---

## 练习题

1. **索引选择**：对于 100 万文档，选择什么索引类型和参数？

2. **分块策略**：比较不同 chunk_size 对检索效果的影响。

3. **评估指标**：如何设计 RAG 系统的端到端评估体系？

---

[← 上一节：4.2 提示词工程](4-2-prompt-engineering.md) | [下一节：4.4 Agent 应用开发 →](4-4-agent.md)
