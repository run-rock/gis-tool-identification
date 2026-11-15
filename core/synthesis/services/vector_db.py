#!/usr/bin/env python3
"""
向量数据库管理 - 封装FAISS向量数据库的加载和搜索
"""

import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from .embedding_service import EmbeddingService


class VectorDatabase:
    """
    向量数据库

    封装FAISS索引的加载、搜索等操作
    与embedding_service配合使用
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        normalize_embeddings: bool = True
    ):
        """
        初始化向量数据库

        Args:
            embedding_service: Embedding服务
            normalize_embeddings: 是否归一化向量
        """
        self.embedding_service = embedding_service
        self.normalize_embeddings = normalize_embeddings

        # FAISS索引和元数据
        self.index: Optional[faiss.Index] = None
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_dim: Optional[int] = None

    def load_from_disk(self, db_path: str):
        """
        从磁盘加载FAISS数据库

        Args:
            db_path: 数据库路径（不含扩展名）
                    例如: /path/to/tools_db
                    会加载: tools_db.faiss 和 tools_db.pkl
        """
        faiss_file = f"{db_path}.faiss"
        pkl_file = f"{db_path}.pkl"

        if not os.path.exists(faiss_file):
            raise FileNotFoundError(f"FAISS索引文件不存在: {faiss_file}")
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"元数据文件不存在: {pkl_file}")

        print(f"📚 加载向量数据库: {db_path}")

        # 加载FAISS索引
        self.index = faiss.read_index(faiss_file)

        # 加载元数据
        with open(pkl_file, 'rb') as f:
            metadata_dict = pickle.load(f)

        self.texts = metadata_dict['texts']
        self.metadata = metadata_dict['metadata']
        self.embedding_dim = metadata_dict['embedding_dim']

        print(f"✅ 向量数据库加载完成")
        print(f"   - 索引数量: {len(self.texts)}")
        print(f"   - 向量维度: {self.embedding_dim}")

    def search(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True
    ) -> Dict[str, Any]:
        """
        搜索相似向量

        Args:
            query: 查询文本
            k: 返回top-k个结果
            return_scores: 是否返回相似度分数

        Returns:
            搜索结果字典，格式为:
            {
                'query': str,  # 原始查询
                'results': [   # 结果列表
                    {
                        'text': str,      # 文本内容
                        'metadata': dict, # 元数据（包含name, algorithm_id, description, parameters等）
                        'score': float    # 相似度分数（可选）
                    },
                    ...
                ]
            }
        """
        if self.index is None:
            raise RuntimeError("向量数据库未加载，请先调用 load_from_disk()")

        # 编码查询文本
        query_embedding = self.embedding_service.encode([query], show_progress=False)

        # 归一化（如果需要）
        if self.normalize_embeddings:
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # FAISS搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS找不到足够的结果时会返回-1
                break

            result = {
                'text': self.texts[idx],
                'metadata': self.metadata[idx]
            }

            if return_scores:
                result['score'] = float(scores[0][i])

            results.append(result)

        return {
            'query': query,
            'results': results
        }

    def search_similar_tools(
        self,
        query: str,
        k: int = 3,
        exclude_tool_id: Optional[str] = None,
        extra_search: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索相似工具（专用方法，处理排除和字段提取）

        Args:
            query: 查询文本（案例文本）
            k: 返回的相似工具数量
            exclude_tool_id: 要排除的工具ID（通常是当前工具）
            extra_search: 额外搜索数量（用于过滤后仍有足够结果）

        Returns:
            相似工具列表，每个工具包含4个字段：
            [
                {
                    'name': str,
                    'algorithm_id': str,
                    'description': str,
                    'parameters': dict
                },
                ...
            ]
        """
        # 搜索更多结果以便过滤后仍有足够数量
        search_results = self.search(query, k=k + extra_search, return_scores=True)

        similar_tools = []
        for result in search_results['results']:
            metadata = result['metadata']
            algorithm_id = metadata.get('algorithm_id', '')

            # 排除当前工具
            if exclude_tool_id and algorithm_id == exclude_tool_id:
                continue

            # 只保留传给大模型需要的4个字段
            similar_tool = {
                'name': metadata.get('name', ''),
                'algorithm_id': algorithm_id,
                'description': metadata.get('description', ''),
                'parameters': metadata.get('parameters', {})
            }
            similar_tools.append(similar_tool)

            # 收集够数量就停止
            if len(similar_tools) >= k:
                break

        return similar_tools

    def batch_search_similar_tools(
        self,
        queries: List[str],
        k: int = 3,
        exclude_tool_id: Optional[str] = None,
        extra_search: int = 10,
        show_progress: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """
        批量搜索相似工具

        Args:
            queries: 查询文本列表
            k: 每个查询返回的相似工具数量
            exclude_tool_id: 要排除的工具ID
            extra_search: 额外搜索数量
            show_progress: 是否显示进度

        Returns:
            相似工具列表的列表
        """
        # 编码所有查询
        query_embeddings = self.embedding_service.encode(queries, show_progress=show_progress)

        # 归一化（如果需要）
        if self.normalize_embeddings:
            query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        # FAISS批量搜索
        scores_batch, indices_batch = self.index.search(
            query_embeddings.astype('float32'),
            k + extra_search
        )

        # 处理每个查询的结果
        all_similar_tools = []
        for scores, indices in zip(scores_batch, indices_batch):
            similar_tools = []

            for idx in indices:
                if idx == -1:
                    break

                metadata = self.metadata[idx]
                algorithm_id = metadata.get('algorithm_id', '')

                # 排除当前工具
                if exclude_tool_id and algorithm_id == exclude_tool_id:
                    continue

                # 只保留传给大模型需要的4个字段
                similar_tool = {
                    'name': metadata.get('name', ''),
                    'algorithm_id': algorithm_id,
                    'description': metadata.get('description', ''),
                    'parameters': metadata.get('parameters', {})
                }
                similar_tools.append(similar_tool)

                # 收集够数量就停止
                if len(similar_tools) >= k:
                    break

            all_similar_tools.append(similar_tools)

        return all_similar_tools

    def __len__(self) -> int:
        """返回数据库中的条目数量"""
        return len(self.texts)

    def __repr__(self) -> str:
        """字符串表示"""
        if self.index is None:
            return "VectorDatabase(未加载)"
        return f"VectorDatabase(条目数={len(self.texts)}, 维度={self.embedding_dim})"
