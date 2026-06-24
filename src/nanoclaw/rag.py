# rag.py
# ======
# RAG (Retrieval-Augmented Generation) 知识库模块。
#
# 技术栈: ChromaDB (向量存储) + sentence-transformers (embedding)
#
# 核心流程:
#   1. 文档摄入 (ingest): 读取 knowledge/ 目录 → 分块 → 向量化 → 存入 ChromaDB
#   2. 语义检索 (query):  用户问题 → 向量化 → ChromaDB 搜索 → 返回 top-K
#
# 为什么用 ChromaDB 而不是 FAISS？
#   ChromaDB 自带持久化、元数据过滤，且 Python 原生，不需要额外服务进程。
#   对小规模个人知识库（几百到几千文档）完全够用。
#
# 为什么用 BAAI/bge-small-zh-v1.5？
#   中文语义检索的性价比之王：384 维向量，模型仅 100MB，
#   在 MTEB 中文榜单上表现优异，CPU 推理也很快。

from __future__ import annotations

# 用 pysqlite3-binary 替换系统自带的旧版 sqlite3（服务器 sqlite 3.31 < ChromaDB 要求的 3.35）
# 必须在任何 chromadb import 之前执行
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 国内服务器无法直连 HuggingFace，使用镜像站下载模型
import os as _os
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chromadb.api import Collection
    from sentence_transformers import SentenceTransformer

from nanoclaw.config import (
    CHROMA_DB_PATH,
    KNOWLEDGE_DIR,
    config,
)

logger = logging.getLogger(__name__)

# 分块参数
_CHUNK_SIZE = 500       # 每块最大字符数
_CHUNK_OVERLAP = 50     # 相邻块之间重叠的字符数

# 检索参数
_TOP_K = 5              # 每次查询返回的最相关结果数

# ChromaDB collection 名称
_COLLECTION_NAME = "knowledge_base"

# 支持的文档格式
_SUPPORTED_SUFFIXES = {".txt", ".md"}

# 模块级单例
_rag_instance: RAG | None = None

# embedding device
_embedding_device: str = "cuda:0"


def get_rag() -> RAG:
    """获取 RAG 引擎的单例实例（延迟初始化）。"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance


class RAG:
    """RAG 知识库引擎。

    负责文档分块、向量化存储和语义检索。
    """

    def __init__(self) -> None:
        self._embedding_model: SentenceTransformer | None = None
        self._chroma_client: Any = None
        self._collection: Collection | None = None

    # ---- 延迟初始化 ----

    @property
    def embedding_model(self) -> SentenceTransformer:
        """延迟加载 embedding 模型（首次使用时才下载和加载）。"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = config.rag_embedding_model
            logger.info("Loading embedding model: %s ...", model_name)
            self._embedding_model = SentenceTransformer(model_name, device=_embedding_device)
            logger.info("Embedding model loaded (dim=%d)", self._embedding_model.get_sentence_embedding_dimension())
        return self._embedding_model

    @property
    def collection(self) -> Collection:
        """延迟初始化 ChromaDB collection。"""
        if self._collection is None:
            import chromadb
            CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            self._collection = self._chroma_client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB collection ready: %s (%d docs)",
                        _COLLECTION_NAME, self._collection.count())
        return self._collection

    # ---- 文档分块 ----

    @staticmethod
    def _chunk_text(text: str, source: str) -> list[tuple[str, dict]]:
        """将文本切分成带重叠的块。

        策略：
        1. 先按段落（双换行）切分，保持语义完整性
        2. 如果单段超过 _CHUNK_SIZE，再按字符滑动窗口切分
        3. 相邻块之间保留 _CHUNK_OVERLAP 的重叠，避免关键信息被截断

        Returns:
            [(chunk_text, metadata), ...]
        """
        paragraphs = text.split("\n\n")
        chunks: list[tuple[str, dict]] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= _CHUNK_SIZE:
                chunks.append((para, {"source": source}))
            else:
                # 长段落：滑动窗口切分
                start = 0
                while start < len(para):
                    end = min(start + _CHUNK_SIZE, len(para))
                    chunk = para[start:end].strip()
                    if chunk:
                        chunks.append((chunk, {"source": source}))
                    start += _CHUNK_SIZE - _CHUNK_OVERLAP

        return chunks

    # ---- 文档摄入 ----

    def ingest_directory(self) -> dict[str, int] | None:
        """扫描 knowledge/ 目录，摄入所有支持的文档。

        对每个文件：
        1. 读取内容
        2. 分块
        3. 向量化
        4. 存入 ChromaDB（覆盖该文件的旧向量）

        Returns:
            {filename: chunk_count} 或 None（没有找到支持的文档）
        """
        files = sorted(
            f for f in KNOWLEDGE_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
        )

        if not files:
            logger.info("No supported files found in %s", KNOWLEDGE_DIR)
            return None

        # 删除旧数据，全量重建（简单且可靠，适合个人知识库规模）
        self._reset_collection()

        stats: dict[str, int] = {}
        all_chunks: list[str] = []
        all_ids: list[str] = []
        all_metadatas: list[dict] = []
        chunk_idx = 0

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("Skipping non-UTF-8 file: %s", file_path.name)
                continue

            source = file_path.name
            chunks = self._chunk_text(content, source)

            if not chunks:
                continue

            for text, meta in chunks:
                all_chunks.append(text)
                all_ids.append(f"{source}:{chunk_idx}")
                all_metadatas.append(meta)
                chunk_idx += 1

            stats[source] = len(chunks)
            logger.info("Ingested %s: %d chunks", source, len(chunks))

        if not all_chunks:
            return None

        # 批量向量化 + 存储
        logger.info("Embedding %d chunks ...", len(all_chunks))
        embeddings = self.embedding_model.encode(
            all_chunks,
            show_progress_bar=False,
            normalize_embeddings=True,  # 归一化，配合 cosine 距离
        )

        self.collection.add(
            ids=all_ids,
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadatas,
        )

        logger.info("Ingestion complete: %d chunks from %d file(s)", len(all_chunks), len(stats))
        return stats

    # ---- 语义检索 ----

    def query(self, query: str) -> list[str]:
        """对知识库执行语义检索。

        流程: query → embedding → ChromaDB 搜索 top-K → 格式化返回

        Returns:
            格式化后的文本块列表，每个元素包含来源文件和内容。
            如果没有结果，返回空列表。
        """
        if self.collection.count() == 0:
            logger.warning("Query on empty collection")
            return []

        # 向量化查询
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )

        # ChromaDB 检索
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=_TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        # 格式化返回
        formatted: list[str] = []
        if not results["ids"] or not results["ids"][0]:
            return []

        for i, doc_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            source = meta.get("source", "unknown")

            # cosine 距离转相似度分数 (0~1, 越高越相关)
            similarity = 1 - distance

            formatted.append(
                f"[{source}] (relevance: {similarity:.2f})\n{doc}"
            )

        return formatted

    # ---- 内部方法 ----

    def _reset_collection(self) -> None:
        """清空并重建 collection。"""
        try:
            # 确保 client 已初始化
            if self._chroma_client is None:
                self.collection  # 触发初始化
            self._chroma_client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
