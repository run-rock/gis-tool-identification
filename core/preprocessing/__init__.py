"""
训练数据预处理模块

包含三个阶段：
1. 数据合并、打乱、分割（splitter）
2. 构建6种类型的样本（builders）
3. JSONL到CSV格式转换（converter）
"""

from .config import PreprocessConfig
from .splitter import DataSplitter
from .converter import TripletConverter, ConversionConfig, CONVERSION_CONFIGS
from . import formatters

__all__ = [
    'PreprocessConfig',
    'DataSplitter',
    'TripletConverter',
    'ConversionConfig',
    'CONVERSION_CONFIGS',
    'formatters',
]
