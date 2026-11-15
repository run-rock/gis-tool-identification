#!/usr/bin/env python3
"""
统一的向量数据库类
基于FAISS实现，支持所有embedding模型
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Union, Optional
from core.common.embedding_model import BaseEmbeddingModel


class FAISSVectorDatabase:
    """统一的FAISS向量数据库"""

    def __init__(self, embedding_model: BaseEmbeddingModel,
                 index_type: str = "IndexFlatIP",
                 normalize_embeddings: bool = True):
        """
        初始化FAISS向量数据库

        Args:
            embedding_model: embedding模型实例
            index_type: FAISS索引类型
            normalize_embeddings: 是否归一化embedding
        """
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.normalize_embeddings = normalize_embeddings

        # 获取embedding维度
        test_embedding = self.embedding_model.encode(["test"])
        self.embedding_dim = test_embedding.shape[1]

        # 初始化FAISS索引
        self.index = None
        self.texts = []
        self.metadata = []

        self._create_index()

        print(f"✅ FAISS向量数据库初始化完成")
        print(f"   索引类型: {index_type}")
        print(f"   嵌入维度: {self.embedding_dim}")
        print(f"   归一化: {normalize_embeddings}")

    def _create_index(self):
        """创建FAISS索引"""
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

    def add_texts(self, texts: List[str],
                  metadata: Optional[List[Dict]] = None,
                  batch_size: int = 32,
                  show_progress: bool = True):
        """
        添加文本到向量数据库

        Args:
            texts: 文本列表
            metadata: 元数据列表
            batch_size: 编码批大小
            show_progress: 是否显示进度
        """
        print(f"🔄 添加 {len(texts)} 个文本到向量数据库...")

        # 去重
        unique_texts = list(dict.fromkeys(texts))
        print(f"   去重后: {len(unique_texts)} 个文本")

        # 编码文本
        embeddings = self.embedding_model.encode(
            unique_texts,
            batch_size=batch_size,
            normalize=self.normalize_embeddings
        )

        # 添加向量到索引
        self.index.add(embeddings.astype(np.float32))

        # 存储文本和元数据
        self.texts.extend(unique_texts)
        if metadata is None:
            metadata = [{"id": len(self.metadata) + i} for i in range(len(unique_texts))]
        self.metadata.extend(metadata)

        print(f"✅ 成功添加，当前总数: {len(self.texts)}")

    def search(self, query: Union[str, List[str]],
               k: int = 5) -> Union[Dict, List[Dict]]:
        """
        搜索相似文本

        Args:
            query: 查询文本或文本列表
            k: 返回top-k个结果

        Returns:
            搜索结果字典或字典列表
        """
        # 处理输入
        single_query = isinstance(query, str)
        if single_query:
            query = [query]

        # 编码查询（多线程环境下禁用进度条避免竞态条件）
        query_embeddings = self.embedding_model.encode(
            query,
            normalize=self.normalize_embeddings,
            show_progress=False
        )

        # 搜索
        scores, indices = self.index.search(query_embeddings.astype(np.float32), k)

        # 构建结果
        results = []
        for query_text, query_scores, query_indices in zip(query, scores, indices):
            query_results = {
                "query": query_text,
                "results": []
            }

            for score, idx in zip(query_scores, query_indices):
                if idx != -1:
                    result = {
                        "text": self.texts[idx],
                        "score": float(score),
                        "metadata": self.metadata[idx],
                        "index": int(idx)
                    }
                    query_results["results"].append(result)

            results.append(query_results)

        return results[0] if single_query else results

    def save(self, save_path: str):
        """
        保存向量数据库到文件

        Args:
            save_path: 保存路径（不包含扩展名）
        """
        print(f"💾 保存向量数据库到: {save_path}")

        # 保存FAISS索引（使用.faiss格式）
        index_path = f"{save_path}.faiss"
        faiss.write_index(self.index, index_path)

        # 保存文本和元数据
        data_path = f"{save_path}.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "index_type": self.index_type,
                "embedding_dim": self.embedding_dim,
                "normalize_embeddings": self.normalize_embeddings
            }, f)

        print(f"✅ 保存完成")
        print(f"   索引文件: {index_path}")
        print(f"   数据文件: {data_path}")

    def load(self, save_path: str):
        """
        从文件加载向量数据库

        Args:
            save_path: 保存路径（不包含扩展名）
        """
        print(f"📂 从文件加载向量数据库: {save_path}")

        # 优先尝试 .faiss 格式，如果不存在再尝试 .index 格式
        faiss_path = f"{save_path}.faiss"
        index_path = f"{save_path}.index"

        if os.path.exists(faiss_path):
            print(f"   检测到 .faiss 格式文件")
            self.index = faiss.read_index(faiss_path)
        elif os.path.exists(index_path):
            print(f"   检测到 .index 格式文件")
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"索引文件不存在: 未找到 {faiss_path} 或 {index_path}")

        # 加载文本和元数据
        data_path = f"{save_path}.pkl"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.texts = data["texts"]
        self.metadata = data["metadata"]
        self.index_type = data["index_type"]
        self.embedding_dim = data["embedding_dim"]
        self.normalize_embeddings = data["normalize_embeddings"]

        print(f"✅ 加载完成")
        print(f"   文本数量: {len(self.texts)}")
        print(f"   索引类型: {self.index_type}")
        print(f"   嵌入维度: {self.embedding_dim}")

    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        return {
            "total_texts": len(self.texts),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "normalize_embeddings": self.normalize_embeddings,
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else 0
        }
