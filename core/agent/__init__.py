"""
Agent核心模块

提供工具识别智能体的核心功能：
- embedding_model: Embedding模型封装（Base/LoRA/API）- 从core.common重导出
- vector_db: FAISS向量数据库
- data_parser: 数据格式化工具
- prompts: 提示词模板
- toolbox_tool_matcher_3: 工具识别器v3
"""

from core.common.embedding_model import (
    BaseEmbeddingModel,
    QwenBaseModel,
    QwenLoRAModel,
    APIEmbeddingModel,
    create_embedding_model
)
from .vector_db import FAISSVectorDatabase
from .data_parser import (
    format_tool_info,
    format_toolbox_info,
    extract_tools_and_toolboxes,
    build_algorithm_to_toolbox_mapping,
    extract_cases_from_test_file
)
from .toolbox_tool_matcher_3 import ToolMatcher

__all__ = [
    # Embedding模型
    'BaseEmbeddingModel',
    'QwenBaseModel',
    'QwenLoRAModel',
    'APIEmbeddingModel',
    'create_embedding_model',
    # 向量数据库
    'FAISSVectorDatabase',
    # 数据解析
    'format_tool_info',
    'format_toolbox_info',
    'extract_tools_and_toolboxes',
    'build_algorithm_to_toolbox_mapping',
    'extract_cases_from_test_file',
    # 工具识别器
    'ToolMatcher',
]
