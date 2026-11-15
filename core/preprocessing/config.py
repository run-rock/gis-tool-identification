"""
预处理配置类
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessConfig:
    """
    训练数据预处理配置

    Attributes:
        random_seed: 随机种子，用于数据打乱和负样本采样（默认42）
        train_ratio: 训练集比例（默认0.8，即4:1划分）

        # 阶段2：样本构建配置
        random_negatives_per_positive: 案例类样本的随机负样本数量（默认2）
        tool_toolbox_negatives: 工具-工具箱样本的负样本数量（默认5）
        toolbox_tool_negatives: 工具箱-工具样本的负样本数量（默认5）

        # 工具文件路径
        tools_file: 工具数据文件路径
    """
    # 通用配置
    random_seed: int = 42

    # 阶段1：分割配置
    train_ratio: float = 0.8

    # 阶段2：样本构建配置
    random_negatives_per_positive: int = 2  # 案例类样本（案例-工具、案例-工具箱、工具-案例、工具箱-案例）
    tool_toolbox_negatives: int = 5         # 工具-工具箱样本
    toolbox_tool_negatives: int = 5         # 工具箱-工具样本

    # 工具文件路径（默认路径）
    tools_file: str = "casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json"
