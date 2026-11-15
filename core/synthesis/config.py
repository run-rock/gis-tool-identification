#!/usr/bin/env python3
"""
配置管理 - 统一的配置系统


  | 维度    | models.py (领域模型)     | config.py (配置模型)       |
  |-------|----------------------|------------------------|
  | 核心职责  | 描述业务领域的概念            | 描述系统配置和运行参数            |
  | 数据来源  | 业务数据（用户生成、LLM生成、数据库） | 配置文件/环境变量/用户设置         |
  | 生命周期  | 短暂（每个请求/任务不同）        | 长期（启动时加载，全局共享）         |
  | 变化频率  | 频繁变化（每次运行都不同）        | 很少变化（只在部署/配置时改）        |
  | 用途    | 业务逻辑处理               | 控制程序行为                 |
  | 序列化目标 | 保存业务结果（JSON文件、数据库）   | 保存配置（config.yaml、.env） |

"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class ModelType(Enum):
    """Embedding模型类型"""
    BASE = "base"    # 基础模型（全精度，以FP16加载）
    LORA = "lora"    # LoRA微调模型（全精度基模型 + LoRA）


@dataclass
class EmbeddingConfig:
    """
    Embedding配置（本地模型）

    使用你的原始QwenBaseModel和QwenLoRAModel加载方式：
    - BASE: 全精度基模型，加载时转FP16
    - LORA: 全精度基模型 + LoRA权重，加载时转FP16
    """
    model_type: ModelType = ModelType.BASE
    base_model_path: str = "models/Qwen3-Embedding-8b"  # 默认全精度基模型
    lora_ckpt_path: Optional[str] = None  # LoRA checkpoint路径（可选）
    pooling_method: str = "last_token"  # last_token 或 mean
    device: str = "cuda:0"
    batch_size: int = 32
    max_length: int = 8192
    normalize: bool = True  # 是否归一化

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_type': self.model_type.value,
            'base_model_path': self.base_model_path,
            'lora_ckpt_path': self.lora_ckpt_path,
            'pooling_method': self.pooling_method,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'normalize': self.normalize,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingConfig':
        """从字典创建"""
        return cls(
            model_type=ModelType(data.get('model_type', 'base')),
            base_model_path=data.get('base_model_path'),
            lora_ckpt_path=data.get('lora_ckpt_path'),
            pooling_method=data.get('pooling_method', 'last_token'),
            device=data.get('device', 'cuda:0'),
            batch_size=data.get('batch_size', 32),
            max_length=data.get('max_length', 8192),
            normalize=data.get('normalize', True),
        )

    def validate(self):
        """验证配置"""
        if not self.base_model_path:
            raise ValueError("必须指定 base_model_path")
        if self.model_type == ModelType.LORA and not self.lora_ckpt_path:
            raise ValueError("LoRA模型必须指定 lora_ckpt_path")


@dataclass
class LLMConfig:
    """
    LLM配置（统一使用OpenAI兼容接口）
    """
    model: str = "qwen3-32b-gguf"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model': self.model,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """从字典创建"""
        return cls(
            model=data.get('model', 'qwen3-32b-gguf'),
            api_key=data.get('api_key'),
            base_url=data.get('base_url'),
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens', 4096),
            timeout=data.get('timeout', 60),
        )

    @classmethod
    def from_env(cls, model_name: str = "qwen3-32b-gguf") -> 'LLMConfig':
        """从环境变量创建"""
        return cls(
            model=model_name,
            api_key=os.getenv("QWEN3_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("QWEN3_URL") or os.getenv("OPENAI_BASE_URL"),
        )

    def validate(self):
        """验证配置"""
        if not self.base_url:
            raise ValueError("LLM必须配置 base_url")


@dataclass
class ProcessConfig:
    """处理配置"""
    max_workers: int = 50
    retry_attempts: int = 2
    retry_delay: int = 300  # API失败重试延迟时间(秒)
    enable_retry: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'max_workers': self.max_workers,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'enable_retry': self.enable_retry,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessConfig':
        """从字典创建"""
        return cls(
            max_workers=data.get('max_workers', 50),
            retry_attempts=data.get('retry_attempts', 2),
            retry_delay=data.get('retry_delay', 300),
            enable_retry=data.get('enable_retry', True),
        )


@dataclass
class AppConfig:
    """应用总配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'embedding': self.embedding.to_dict(),
            'llm': self.llm.to_dict(),
            'process': self.process.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """从字典创建"""
        return cls(
            embedding=EmbeddingConfig.from_dict(data.get('embedding', {})),
            llm=LLMConfig.from_dict(data.get('llm', {})),
            process=ProcessConfig.from_dict(data.get('process', {})),
        )

    @classmethod
    def default_base_model(cls) -> 'AppConfig':
        """默认配置：基础模型（全精度，FP16加载）"""
        return cls(
            embedding=EmbeddingConfig.base_model(
                model_path="models/Qwen3-Embedding-8b"
            ),
            llm=LLMConfig.from_env()
        )

    @classmethod
    def default_lora_model(cls, checkpoint: str = "checkpoint-3200") -> 'AppConfig':
        """
        默认配置：LoRA模型（全精度基模型 + LoRA）

        Args:
            checkpoint: checkpoint名称，如 "checkpoint-3200"
        """
        return cls(
            embedding=EmbeddingConfig.lora_model(
                base_model_path="models/Qwen3-Embedding-8b",
                lora_ckpt_path=f"models/sftmodel/qwen3-embedding-8b/1/2025-10-13_21-01-27/{checkpoint}"
            ),
            llm=LLMConfig.from_env()
        )

    def validate(self):
        """验证配置"""
        self.embedding.validate()
        self.llm.validate()

    def print_summary(self):
        """打印配置摘要"""
        print("=" * 80)
        print("应用配置")
        print("=" * 80)
        print(f"\n🔧 Embedding配置:")
        print(f"  类型: {self.embedding.model_type.value}")
        print(f"  基模型: {self.embedding.base_model_path}")
        if self.embedding.model_type == ModelType.LORA:
            print(f"  LoRA: {self.embedding.lora_ckpt_path}")
        print(f"  设备: {self.embedding.device}")
        print(f"  池化: {self.embedding.pooling_method}")
        print(f"  批次: {self.embedding.batch_size}")

        print(f"\n🤖 LLM配置:")
        print(f"  模型: {self.llm.model}")
        print(f"  API URL: {self.llm.base_url}")
        print(f"  温度: {self.llm.temperature}")

        print(f"\n⚙️  处理配置:")
        print(f"  最大并发: {self.process.max_workers}")
        print(f"  重试次数: {self.process.retry_attempts}")
        print("=" * 80)
