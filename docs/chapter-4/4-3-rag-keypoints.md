# 4.3 RAG 应用开发 - 关键知识点详解

> 本节为 4.3 节的补充知识点，包含架构选择器、向量数据库对比、检索优化技巧。

---

## 知识点 1: RAG 架构选择器

### RAG 架构演变

```
RAG 架构决策树
│
├── 数据规模？
│   │
│   ├── < 10K 文档
│   │   └── → 简单向量检索 (Chroma/FAISS)
│   │
│   ├── 10K - 1M 文档
│   │   └── → 混合检索 + 重排序 (Elasticsearch + Vector DB)
│   │
│   └── > 1M 文档
│       └── → 分布式向量数据库 (Milvus/Weaviate/Pinecone)
│
├── 延迟要求？
│   │
│   ├── < 100ms
│   │   └── → 向量索引 + 缓存 + 小上下文窗口
│   │
│   ├── 100ms - 1s
│   │   └── → 混合检索 + 中等重排序
│   │
│   └── > 1s
│       └── → 完整 RAG 流程 (检索 + 重排序 + 融合)
│
└── 准确性要求？
    │
    ├── 高准确性
    │   └── → 多路召回 + Cross-Encoder 重排序
    │
    ├── 中等准确性
    │   └── → 混合检索 + MM Rerank
    │
    └── 一般准确性
        └── → 单一向量检索
```

---

## 知识点 2: 向量数据库对比

### 主流向量数据库对比表

| 特性 | Chroma | FAISS | Milvus | Pinecone | Weaviate |
|------|--------|-------|--------|----------|----------|
| **类型** | 嵌入式 | 库 | 服务 | SaaS | 服务 |
| **规模** | < 1M | < 100M | > 1B | > 1B | > 100M |
| **索引类型** | HNSW | IVF/HNSW | IVF/HNSW | 专有 | HNSW |
| **元数据过滤** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **持久化** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **分布式** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **开源** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **适用场景** | 原型/小规模 | 本地推理 | 大规模生产 | 托管服务 |  pengetahuan graph |

### Chroma 快速入门

```python
# examples/4-3-rag/chroma_quickstart.py
"""
Chroma 向量数据库快速入门

安装：pip install chromadb
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict
import hashlib


class ChromaRAG:
    """
    基于 Chroma 的 RAG 系统
    
    功能:
    1. 文档 ingestion (自动分块 + 嵌入)
    2. 相似性检索
    3. 元数据过滤
    """
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        # 持久化客户端
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 余弦相似度
        )
        
        # 嵌入模型 (使用 sentence-transformers)
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """生成嵌入"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None,
        chunk_size: int = 500
    ):
        """
        添加文档到向量库
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表
            ids: 文档 ID (默认使用哈希)
            chunk_size: 分块大小 (字符)
        """
        
        # 文档分块
        chunks = []
        chunk_metadatas = []
        chunk_ids = []
        
        for i, doc in enumerate(documents):
            # 简单分块 (实际应使用语义分块)
            for j in range(0, len(doc), chunk_size):
                chunk = doc[j:j+chunk_size]
                chunks.append(chunk)
                
                # 元数据
                meta = metadatas[i] if metadatas else {}
                meta["source"] = f"doc_{i}"
                meta["chunk_idx"] = j // chunk_size
                chunk_metadatas.append(meta)
                
                # ID (使用内容哈希)
                chunk_id = hashlib.md5(chunk.encode()).hexdigest()
                chunk_ids.append(chunk_id)
        
        # 生成嵌入
        embeddings = self._embed(chunks)
        
        # 添加到 Chroma
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        print(f"添加了 {len(chunks)} 个文档块")
    
    def query(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None
    ) -> Dict:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件
        
        Returns:
            {
                "documents": [...],
                "metadatas": [...],
                "distances": [...]
            }
        """
        
        # 生成查询嵌入
        query_embedding = self._embed([query])[0]
        
        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    
    def hybrid_search(
        self,
        query: str,
        keyword_filter: str = None,
        n_results: int = 10
    ) -> Dict:
        """
        混合检索：关键词 + 向量
        
        先用关键词过滤，再向量检索
        """
        
        # 如果有关键词过滤
        if keyword_filter:
            where = {"source": {"$contains": keyword_filter}}
        else:
            where = None
        
        return self.query(query, n_results=n_results, where=where)


# 使用示例
if __name__ == "__main__":
    rag = ChromaRAG()
    
    # 添加文档
    documents = [
        "人工智能是模拟人类智能的科学。",
        "深度学习是机器学习的一个子领域，使用神经网络。",
        "自然语言处理让计算机理解和生成人类语言。",
        "计算机视觉让计算机能够'看到'和理解图像。"
    ]
    
    rag.add_documents(
        documents=documents,
        metadatas=[{"category": "AI"} for _ in documents]
    )
    
    # 检索
    results = rag.query("什么是深度学习？", n_results=2)
    
    print("检索结果:")
    for doc, dist in zip(results["documents"], results["distances"]):
        print(f"  相似度：{1-dist:.4f} - {doc}")
```

---

## 知识点 3: 检索优化技巧

### 检索增强策略对比

| 策略 | 描述 | 实现难度 | 效果提升 |
|------|------|---------|---------|
| **多查询检索** | 生成多个相关问题并行检索 | ⭐⭐ | +15-25% |
| **假设文档嵌入 (HyDE)** | 生成假设答案再检索 | ⭐⭐⭐ | +20-30% |
| **重排序** | 对检索结果二次排序 | ⭐⭐ | +15-25% |
| **检索 - 阅读器微调** | 端到端优化 | ⭐⭐⭐⭐ | +25-35% |

### HyDE (Hypothetical Document Embeddings) 实现

```python
# examples/4-3-rag/hyde_retrieval.py
"""
HyDE: Hypothetical Document Embeddings

参考：Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels"

核心思想：
1. LLM 生成假设答案
2. 嵌入假设答案 (而非原始查询)
3. 用假设答案的嵌入检索真实文档
"""

from typing import List
from openai import OpenAI


class HyDERetriever:
    """
    HyDE 检索器
    
    相比普通检索的优势：
    - 假设答案与真实文档在向量空间更接近
    - 减少查询 - 文档的语义鸿沟
    """
    
    def __init__(
        self,
        vector_store,  # 向量存储接口
        llm_model: str = "gpt-3.5-turbo",
        max_tokens: int = 200
    ):
        self.vector_store = vector_store
        self.llm = OpenAI()
        self.llm_model = llm_model
        self.max_tokens = max_tokens
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        生成假设答案
        
        Prompt 关键：
        - 不要求答案正确，只需"像"真实文档
        - 鼓励使用与目标文档相似的文体
        """
        
        prompt = f"""请为以下问题写一个假设性答案。
答案不需要正确，但应该看起来像是从相关文档中摘录的。
使用正式、信息性的语言风格。

问题：{query}

假设答案："""
        
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        HyDE 检索流程
        
        1. 生成假设答案
        2. 嵌入假设答案
        3. 检索相似文档
        """
        
        # Step 1: 生成假设答案
        hypothetical_answer = self.generate_hypothetical_answer(query)
        print(f"假设答案：{hypothetical_answer[:100]}...")
        
        # Step 2-3: 用假设答案检索 (向量库内部会嵌入)
        results = self.vector_store.query(
            query=hypothetical_answer,
            n_results=n_results
        )
        
        # 可选：用原始查询再检索一次，然后融合
        # 这叫"查询扩展"或"多向量检索"
        
        return results


# 对比实验
def compare_retrieval_methods(query: str):
    """比较普通检索 vs HyDE"""
    
    # 普通检索
    normal_results = vector_store.query(query, n_results=5)
    
    # HyDE 检索
    hyde = HyDERetriever(vector_store)
    hyde_results = hyde.retrieve(query, n_results=5)
    
    print("\n=== 普通检索结果 ===")
    for doc in normal_results["documents"][:3]:
        print(doc[:100])
    
    print("\n=== HyDE 检索结果 ===")
    for doc in hyde_results["documents"][:3]:
        print(doc[:100])


if __name__ == "__main__":
    # 需要先初始化 vector_store
    # compare_retrieval_methods("如何优化 Transformer 的推理速度？")
    pass
```

### 重排序 (Reranking) 实现

```python
# examples/4-3-rag/reranking.py
"""
检索结果重排序

方法对比:
1. Cross-Encoder (最准确，最慢)
2. LLM Rerank (灵活，中等速度)
3. Rule-based (最快，准确性一般)
"""

from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    """
    检索结果重排序器
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5
    ):
        """
        Args:
            model_name: Cross-Encoder 模型
            top_k: 返回前 k 个结果
        """
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: List[float] = None
    ) -> List[Tuple[int, str, float]]:
        """
        重排序文档
        
        Args:
            query: 查询
            documents: 候选文档列表
            scores: 初始检索分数 (可选，用于融合)
        
        Returns:
            [(doc_idx, doc_text, final_score), ...]
        """
        
        # 构建 (query, doc) 对
        pairs = [[query, doc] for doc in documents]
        
        # Cross-Encoder 打分
        ce_scores = self.model.predict(pairs)
        
        # 如果有初始分数，可以融合
        if scores is not None:
            # 归一化
            ce_scores_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            # 融合 (加权平均)
            final_scores = 0.7 * ce_scores_norm + 0.3 * scores_norm
        else:
            final_scores = ce_scores
        
        # 排序
        indices = np.argsort(final_scores)[::-1][:self.top_k]
        
        return [
            (idx, documents[idx], float(final_scores[idx]))
            for idx in indices
        ]


class LLMReranker:
    """
    使用 LLM 进行重排序
    
    优势：
    - 可以理解复杂的相关性
    - 可以加入领域知识
    - 可以解释排序理由
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model = model
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        n_results: int = 3
    ) -> List[str]:
        """
        使用 LLM 对文档进行相关性排序
        """
        
        # 构建提示
        doc_list = "\n\n".join(
            f"[文档 {i}]\n{doc}"
            for i, doc in enumerate(documents)
        )
        
        prompt = f"""请评估以下文档与查询的相关性，并按相关性从高到低排序。

查询：{query}

{doc_list}

请只返回排序后的文档编号，用逗号分隔，例如：2,0,1,3
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # 解析结果
        try:
            indices = [int(x.strip()) for x in response.choices[0].message.content.split(",")]
            return [documents[i] for i in indices if i < len(documents)][:n_results]
        except:
            return documents[:n_results]  # 降级处理


# 使用示例
if __name__ == "__main__":
    # Cross-Encoder 重排序
    reranker = Reranker()
    
    query = "如何训练大语言模型？"
    documents = [
        "深度学习需要大量的数据和计算资源。",
        "训练大语言模型包括预训练和微调两个阶段。",
        "今天的天气很好。",  # 无关文档
        "语言模型使用 Transformer 架构，通过下一个词预测任务进行训练。"
    ]
    
    # 初始检索分数 (模拟)
    initial_scores = [0.5, 0.7, 0.2, 0.6]
    
    # 重排序
    ranked = reranker.rerank(query, documents, initial_scores)
    
    print("重排序结果:")
    for idx, doc, score in ranked:
        print(f"  {score:.3f}: {doc[:50]}...")
```

---

## 知识点 4: 完整 RAG 系统示例

```python
# examples/4-3-rag/full_rag_system.py
"""
完整 RAG 系统示例

架构:
1. 文档处理 → 2. 向量化 → 3. 检索 → 4. 重排序 → 5. 生成
"""

import os
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class RAGConfig:
    """RAG 配置"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gpt-3.5-turbo"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieve: int = 10
    top_k_rerank: int = 5


class RAGSystem:
    """
    完整 RAG 系统
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = OpenAI()
        
        # 初始化组件
        self._init_embedding()
        self._init_reranker()
        self._init_vector_store()
    
    def _init_embedding(self):
        """初始化嵌入模型"""
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.config.embedding_model)
    
    def _init_reranker(self):
        """初始化重排序模型"""
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(self.config.rerank_model)
    
    def _init_vector_store(self):
        """初始化向量存储"""
        import chromadb
        self.vector_store = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.vector_store.get_or_create_collection("rag_docs")
    
    def _chunk_documents(self, documents: List[str]) -> List[Dict]:
        """文档分块"""
        chunks = []
        
        for doc_idx, doc in enumerate(documents):
            # 简单字符分块 (实际应使用语义分块)
            for i in range(0, len(doc), self.config.chunk_size - self.config.chunk_overlap):
                chunk = doc[i:i + self.config.chunk_size]
                chunks.append({
                    "id": f"doc_{doc_idx}_chunk_{i}",
                    "content": chunk,
                    "metadata": {"doc_idx": doc_idx, "start": i}
                })
        
        return chunks
    
    def ingest(self, documents: List[str], metadatas: List[Dict] = None):
        """
        文档入库
        
        流程：
        1. 分块
        2. 嵌入
        3. 存储
        """
        
        # Step 1: 分块
        chunks = self._chunk_documents(documents)
        print(f"文档分块：{len(documents)} → {len(chunks)} chunks")
        
        # Step 2: 嵌入
        chunk_texts = [c["content"] for c in chunks]
        embeddings = self.embedder.encode(chunk_texts)
        
        # Step 3: 存储
        self.collection.add(
            documents=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=[c["metadata"] for c in chunks],
            ids=[c["id"] for c in chunks]
        )
        
        print(f"文档入库完成")
    
    def retrieve_and_rerank(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        检索 + 重排序
        
        Returns:
            重排序后的文档列表
        """
        
        # Step 1: 嵌入查询
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Step 2: 初始检索
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.config.top_k_retrieve
        )
        
        documents = results["documents"][0]
        scores = results["distances"][0]
        metadatas = results["metadatas"][0]
        
        # Step 3: 重排序
        pairs = [[query, doc] for doc in documents]
        rerank_scores = self.reranker.predict(pairs)
        
        # Step 4: 融合分数
        # 将距离转换为相似度
        similarity_scores = [1 - s for s in scores]
        
        # 归一化
        sim_norm = np.array(similarity_scores) / (sum(similarity_scores) + 1e-8)
        rerank_norm = np.array(rerank_scores) / (sum(rerank_scores) + 1e-8)
        
        # 加权融合
        final_scores = 0.6 * rerank_norm + 0.4 * sim_norm
        
        # Step 5: 排序返回
        indices = np.argsort(final_scores)[::-1][:n_results]
        
        return [
            {
                "content": documents[i],
                "metadata": metadatas[i],
                "score": float(final_scores[i])
            }
            for i in indices
        ]
    
    def generate_answer(
        self,
        query: str,
        context: str,
        temperature: float = 0.7
    ) -> str:
        """
        基于上下文生成答案
        """
        
        prompt = f"""基于以下上下文回答问题。如果上下文中没有答案，请诚实告知。

上下文：
{context}

问题：{query}

答案："""
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": "你是有帮助的助手。基于上下文回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def query(self, query: str) -> Dict:
        """
        完整 RAG 查询流程
        
        Returns:
            {
                "answer": str,
                "sources": [...],
                "retrieval_scores": [...]
            }
        """
        
        # Step 1: 检索 + 重排序
        retrieved = self.retrieve_and_rerank(query, n_results=self.config.top_k_rerank)
        
        # Step 2: 构建上下文
        context = "\n\n".join(
            f"[来源 {i+1}]\n{doc['content']}"
            for i, doc in enumerate(retrieved)
        )
        
        # Step 3: 生成答案
        answer = self.generate_answer(query, context)
        
        return {
            "answer": answer,
            "sources": retrieved,
            "retrieval_scores": [doc["score"] for doc in retrieved]
        }


# 使用示例
if __name__ == "__main__":
    # 初始化
    config = RAGConfig()
    rag = RAGSystem(config)
    
    # 文档入库
    documents = [
        "Transformer 是一种基于自注意力机制的深度学习模型。",
        "BERT 使用 Encoder-only 架构，适合理解型任务。",
        "GPT 使用 Decoder-only 架构，适合生成型任务。"
    ]
    rag.ingest(documents)
    
    # 查询
    result = rag.query("Transformer 和 BERT 有什么区别？")
    
    print(f"答案：{result['answer']}")
    print("\n来源:")
    for src in result['sources']:
        print(f"  {src['score']:.3f}: {src['content'][:50]}...")
```

---

## 练习题

### 练习 1: 设计分块策略

给定以下文档类型，设计合适的分块策略：

| 文档类型 | 特点 | 推荐分块方法 |
|---------|------|------------|
| 技术文档 | 有章节结构 | ? |
| 对话记录 | 自然分段 | ? |
| 代码文件 | 函数/类边界 | ? |

### 练习 2: 评估 RAG 系统

设计评估指标和测试集来评估 RAG 系统的质量。

考虑因素：
- 检索准确性 (Recall@K, MRR)
- 答案质量 (忠实度、准确性)
- 端到端延迟

---

## 延伸阅读

- [RAG 综述论文](https://arxiv.org/abs/2312.10997)
- [HyDE 论文](https://arxiv.org/abs/2212.10496)
- [LangChain RAG 文档](https://python.langchain.com/docs/use_cases/question_answering/)

---

[← 返回 4.3 主文档](4-3-rag.md) | [下一节：Agent 应用开发 →](4-4-agent.md)
