"""
4.3 RAG 应用开发 - 完整 RAG 系统实现

实现内容:
- 向量化与 Embedding
- Chroma 向量数据库
- 检索与重排序
- 完整 RAG 流程
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib


# ========== Embedding 封装 ==========

class EmbeddingModel:
    """
    Embedding 模型封装

    支持:
    - Sentence Transformers
    - OpenAI Embedding API
    - BGE 中文模型
    """

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        if self.model_name.startswith("openai/"):
            # OpenAI API
            from openai import OpenAI
            self.api_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.api_model = self.model_name.replace("openai/", "")
        else:
            # 本地模型
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        向量化文本

        参数:
            texts: 文本列表
            normalize: 是否 L2 归一化

        返回:
            向量矩阵 (n_texts, dim)
        """
        if hasattr(self, 'api_client'):
            # OpenAI API
            response = self.api_client.embeddings.create(
                model=self.api_model,
                input=texts
            )
            embeddings = np.array([item.embedding for item in response.data])
        else:
            # 本地模型
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 10
            )

        return embeddings

    def encode_query(self, text: str) -> np.ndarray:
        """编码查询（某些模型需要特殊前缀）"""
        # BGE 需要添加前缀
        if "bge" in self.model_name.lower():
            text = "为这个句子生成表示以用于检索：" + text
        return self.encode([text])[0]

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """编码文档"""
        if "bge" in self.model_name.lower():
            documents = ["代表这个句子：" + d for d in documents]
        return self.encode(documents)


# ========== 向量数据库 ==========

@dataclass
class Document:
    """文档对象"""
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:16]


class VectorStore:
    """
    向量存储与检索

    基于 ChromaDB
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_fn=None
    ):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_fn = embedding_fn

    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """添加文档"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # 向量化
            if self.embedding_fn:
                texts = [d.content for d in batch]
                embeddings = self.embedding_fn.encode_documents(texts)
            else:
                embeddings = [d.embedding for d in batch if d.embedding is not None]

            self.collection.add(
                documents=[d.content for d in batch],
                embeddings=embeddings if embeddings else None,
                metadatas=[{"doc_id": d.doc_id, **d.metadata} for d in batch],
                ids=[f"{d.doc_id}_{i}" for i, d in enumerate(batch)]
            )

        print(f"已添加 {len(documents)} 个文档")

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None
    ) -> Dict:
        """
        相似度搜索

        返回:
            {
                'documents': [...],
                'metadatas': [...],
                'distances': [...],
                'ids': [...]
            }
        """
        if self.embedding_fn:
            query_embedding = self.embedding_fn.encode_query(query)
        else:
            query_embedding = None

        results = self.collection.query(
            query_embeddings=[query_embedding] if query_embedding else None,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        return results


# ========== 重排序模型 ==========

class Reranker:
    """
    重排序模型

    使用 Cross-Encoder 进行精排
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        重排序

        返回:
            [(原始索引，分数), ...]
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        # 排序
        indexed_scores = list(enumerate(scores))
        ranked = sorted(indexed_scores, key=lambda x: -x[1])[:top_k]

        return ranked


# ========== 完整 RAG 系统 ==========

@dataclass
class RAGConfig:
    """RAG 配置"""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-large"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    rerank_top_k: int = 3


class RAGSystem:
    """
    完整 RAG 系统

    流程:
    1. 文档导入 → 2. 分块 → 3. 向量化 → 4. 存储
    5. 查询 → 6. 检索 → 7. 重排序 → 8. 生成
    """

    def __init__(self, config: RAGConfig):
        self.config = config

        # 初始化组件
        print("初始化 RAG 系统...")
        self.embedder = EmbeddingModel(config.embedding_model)
        self.vector_store = VectorStore(
            embedding_fn=self.embedder
        )

        # 重排序模型（可选）
        if config.rerank_top_k > 0:
            print(f"加载重排序模型：{config.rerank_model}")
            self.reranker = Reranker(config.rerank_model)
        else:
            self.reranker = None

    def _chunk_documents(
        self,
        documents: List[str],
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Document]:
        """文档分块"""
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        chunks = []
        for doc_idx, doc in enumerate(documents):
            start = 0
            while start < len(doc):
                end = start + chunk_size
                chunk_content = doc[start:end]

                if chunk_content.strip():
                    chunks.append(Document(
                        content=chunk_content.strip(),
                        metadata={
                            "doc_idx": doc_idx,
                            "chunk_start": start,
                            "chunk_end": end
                        }
                    ))

                start += chunk_size - overlap

        return chunks

    def ingest(self, documents: List[str]):
        """
        导入文档

        流程:
        1. 分块
        2. 向量化
        3. 存储
        """
        print(f"导入 {len(documents)} 个文档...")

        # 分块
        chunks = self._chunk_documents(documents)
        print(f"分块完成：{len(chunks)} 个块")

        # 向量化并存储
        self.vector_store.add_documents(chunks)
        print("导入完成")

    def query(
        self,
        query: str,
        n_results: int = None,
        use_rerank: bool = True
    ) -> str:
        """
        RAG 查询

        流程:
        1. 检索相关文档
        2. (可选) 重排序
        3. 构建 Prompt
        4. 生成答案

        返回:
            生成的答案
        """
        n_results = n_results or self.config.top_k

        # 1. 检索
        results = self.vector_store.search(
            query=query,
            n_results=n_results * 2 if self.reranker else n_results
        )

        if not results['documents'][0]:
            return "未找到相关信息"

        # 2. 重排序（可选）
        if use_rerank and self.reranker:
            documents = results['documents'][0]

            # 重排序
            ranked = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=self.config.rerank_top_k
            )

            contexts = [documents[i] for i, _ in ranked[:self.config.rerank_top_k]]
        else:
            contexts = results['documents'][0][:n_results]

        # 3. 构建 Prompt
        prompt = self._build_prompt(query, contexts)

        # 4. 生成答案（这里调用 LLM，实际使用时替换为真实调用）
        answer = self._generate_answer(prompt)

        return answer

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        """构建 RAG Prompt"""
        context_text = "\n\n".join(
            f"[文档 {i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        )

        prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请说明。

<documents>
{context_text}
</documents>

问题：{query}

请用中文回答，并引用相关文档的来源。"""

        return prompt

    def _generate_answer(self, prompt: str) -> str:
        """
        生成答案

        实际使用时调用 LLM API
        这里用占位实现
        """
        # 占位实现
        return "[答案将通过调用 LLM API 生成]"

    def stream_query(self, query: str):
        """流式查询（生成器）"""
        # 类似 query 方法，但使用流式 LLM 调用
        yield from self._stream_generate(query)

    def _stream_generate(self, prompt: str):
        """流式生成"""
        # 占位实现
        for token in ["R", "A", "G", " ", "答", "案"]:
            yield token


# ========== 使用示例 ==========

def demo_rag():
    """RAG 系统演示"""

    config = RAGConfig(
        embedding_model="BAAI/bge-large-zh-v1.5",
        top_k=5,
        rerank_top_k=3
    )

    # 创建 RAG 系统
    rag = RAGSystem(config)

    # 导入文档
    documents = [
        "人工智能（AI）是计算机科学的一个分支，研究如何使计算机模拟人类智能。",
        "机器学习是 AI 的核心，通过数据训练模型来实现预测和决策。",
        "深度学习是机器学习的子领域，使用多层神经网络处理复杂任务。",
        "自然语言处理（NLP）研究如何让计算机理解和生成人类语言。",
        "计算机视觉（CV）研究如何让计算机看懂图像和视频。"
    ]

    rag.ingest(documents)

    # 查询
    query = "什么是人工智能？"
    print(f"\n问题：{query}")

    answer = rag.query(query)
    print(f"答案：{answer}")


if __name__ == "__main__":
    demo_rag()
