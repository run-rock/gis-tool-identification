"""
训练模块

包含Embedding模型训练的核心逻辑：
- 数据加载
- 损失函数
- Trainer类
- 工具函数
- 配置记录
"""

from .config import TrainingConfig
from .data_loader import build_triplet_samples, load_training_data, load_validation_data
from .losses import TripletContrastiveLossTrainer
from .utils import last_token_pooling, TripletDataCollator, CombinedLossCallback
from .distance_analyzer import DistanceAnalyzer
from .config_logger import save_training_config

__all__ = [
    'TrainingConfig',
    'build_triplet_samples',
    'load_training_data',
    'load_validation_data',
    'TripletContrastiveLossTrainer',
    'last_token_pooling',
    'TripletDataCollator',
    'CombinedLossCallback',
    'DistanceAnalyzer',
    'save_training_config',
]
