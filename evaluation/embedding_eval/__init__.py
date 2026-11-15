"""
Embedding模型评估模块

包含：
- build_testcases: 测试集构建
- build_all_vectordbs: 一键构建所有向量库（工具、工具箱、案例）
- evaluate_retrieval: 检索评估
- evaluator: 检索评估器（Recall@k, NDCG@k）
"""

from .evaluator import RetrievalEvaluator

__all__ = [
    'RetrievalEvaluator',
]
