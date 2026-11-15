#!/usr/bin/env python3
"""
Embedding服务层 - 封装你原始的QwenBaseModel和QwenLoRAModel
"""

import sys
import os
import numpy as np
from typing import List, Union

# 添加路径以导入embedding_model
sys.path.append('embedding_model_train/toolbench/evaluate/G8')
from core.common.embedding_model import QwenBaseModel, QwenLoRAModel

from ..config import EmbeddingConfig, ModelType


class EmbeddingService:
    """
    Embedding服务

    封装QwenBaseModel和QwenLoRAModel，提供统一接口
    使用你的原始加载方式：全精度模型 + FP16加载
    """

    def __init__(self, config: EmbeddingConfig):
        """
        初始化Embedding服务

        Args:
            config: Embedding配置
        """
        self.config = config

        print("=" * 80)
        print("初始化Embedding服务")
        print("=" * 80)
        print(f"模型类型: {config.model_type.value}")
        print(f"基模型: {config.base_model_path}")

        if config.model_type == ModelType.BASE:
            # 基础模型：全精度，FP16加载
            self.model = QwenBaseModel(
                model_path=config.base_model_path,
                pooling_method=config.pooling_method,
                device=config.device
            )
        elif config.model_type == ModelType.LORA:
            # LoRA模型：全精度基模型 + LoRA权重，FP16加载
            print(f"LoRA: {config.lora_ckpt_path}")
            self.model = QwenLoRAModel(
                base_model_path=config.base_model_path,
                lora_ckpt_path=config.lora_ckpt_path,
                pooling_method=config.pooling_method,
                device=config.device
            )
        else:
            raise ValueError(f"不支持的模型类型: {config.model_type}")

        print("✅ Embedding服务初始化完成")
        print("=" * 80)

    def encode(self, texts: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 单个文本或文本列表
            show_progress: 是否显示进度条

        Returns:
            numpy数组，形状为 (num_texts, embedding_dim)
        """
        return self.model.encode(
            texts=texts,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            show_progress=show_progress,
            normalize=self.config.normalize
        )

    def encode_queries(self, queries: List[str], show_progress: bool = True) -> np.ndarray:
        """
        编码查询文本（便捷方法）

        Args:
            queries: 查询文本列表
            show_progress: 是否显示进度条

        Returns:
            numpy数组
        """
        return self.encode(queries, show_progress=show_progress)

    def encode_corpus(self, corpus: List[str], show_progress: bool = True) -> np.ndarray:
        """
        编码语料库文本（便捷方法）

        Args:
            corpus: 语料库文本列表
            show_progress: 是否显示进度条

        Returns:
            numpy数组
        """
        return self.encode(corpus, show_progress=show_progress)


# ============================================================================
# 便捷函数
# ============================================================================

def create_embedding_service(config: EmbeddingConfig) -> EmbeddingService:
    """
    创建embedding服务（便捷函数）

    Args:
        config: Embedding配置

    Returns:
        EmbeddingService实例
    """
    return EmbeddingService(config)


def create_base_model_service(
    model_path: str = "models/Qwen3-Embedding-8b",
    device: str = "cuda:0",
    pooling_method: str = "last_token",
    batch_size: int = 32,
    max_length: int = 8192,
    normalize: bool = True
) -> EmbeddingService:
    """
    创建基础模型服务（便捷函数）

    Args:
        model_path: 模型路径
        device: 设备
        pooling_method: 池化方式
        batch_size: 批大小
        max_length: 最大长度
        normalize: 是否归一化

    Returns:
        EmbeddingService实例
    """
    config = EmbeddingConfig(
        model_type=ModelType.BASE,
        base_model_path=model_path,
        device=device,
        pooling_method=pooling_method,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize
    )
    return EmbeddingService(config)


def create_lora_model_service(
    base_model_path: str = "models/Qwen3-Embedding-8b",
    lora_ckpt_path: str = "models/sftmodel/qwen3-embedding-8b/1/2025-10-13_21-01-27/checkpoint-3200",
    device: str = "cuda:0",
    pooling_method: str = "last_token",
    batch_size: int = 32,
    max_length: int = 8192,
    normalize: bool = True
) -> EmbeddingService:
    """
    创建LoRA模型服务（便捷函数）

    Args:
        base_model_path: 基模型路径
        lora_ckpt_path: LoRA checkpoint路径
        device: 设备
        pooling_method: 池化方式
        batch_size: 批大小
        max_length: 最大长度
        normalize: 是否归一化

    Returns:
        EmbeddingService实例
    """
    config = EmbeddingConfig(
        model_type=ModelType.LORA,
        base_model_path=base_model_path,
        lora_ckpt_path=lora_ckpt_path,
        device=device,
        pooling_method=pooling_method,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize
    )
    return EmbeddingService(config)
