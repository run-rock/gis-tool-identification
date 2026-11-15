#!/usr/bin/env python3
"""
统一的Embedding模型接口
支持基模型、LoRA模型、API模型的统一调用
"""

import torch
import json
import os
import numpy as np
import time
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from typing import List, Union, Dict
from tqdm import tqdm
from openai import OpenAI


class BaseEmbeddingModel(ABC):
    """Embedding模型抽象基类"""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 单个文本或文本列表
            **kwargs: 其他参数

        Returns:
            numpy数组，形状为 (num_texts, embedding_dim)
        """
        pass


class QwenBaseModel(BaseEmbeddingModel):
    """Qwen基模型实现"""

    def __init__(self, model_path: str, pooling_method: str = "last_token", device: str = "cuda:0"):
        """
        初始化Qwen基模型

        Args:
            model_path: 模型路径
            pooling_method: 池化方法，"last_token" 或 "mean"
            device: 设备
        """
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.device = device

        print("🚀 加载Qwen基模型")
        print("=" * 60)

        # 加载tokenizer
        print("📝 加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.pooling_method == "last_token":
            self.tokenizer.padding_side = 'left'
            print("🔧 设置左对齐padding用于last_token池化")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        print("🤖 加载模型...")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print("✅ 基模型加载完成")

    def last_token_pool(self, last_hidden_states, attention_mask):
        """Last token pooling"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _apply_pooling(self, hidden_states, attention_mask):
        """应用池化"""
        if self.pooling_method == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            return torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_method == "last_token":
            return self.last_token_pool(hidden_states, attention_mask)
        else:
            raise ValueError(f"不支持的pooling方法: {self.pooling_method}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, max_length: int = 8192,
               show_progress: bool = True, normalize: bool = True) -> np.ndarray:
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        pbar = tqdm(range(0, len(texts), batch_size), desc="编码句子") if show_progress else range(0, len(texts), batch_size)

        with torch.no_grad():
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, len(texts))
                batch_sentences = texts[start_idx:end_idx]

                inputs = self.tokenizer(
                    batch_sentences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )

                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

                embeddings = self._apply_pooling(hidden_states, inputs['attention_mask'])

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings


class QwenLoRAModel(BaseEmbeddingModel):
    """Qwen LoRA模型实现"""

    def __init__(self, base_model_path: str, lora_ckpt_path: str, pooling_method: str = "last_token", device: str = "cuda:0"):
        """
        初始化Qwen LoRA模型

        Args:
            base_model_path: 基模型路径
            lora_ckpt_path: LoRA checkpoint路径
            pooling_method: 池化方法
            device: 设备
        """
        self.base_model_path = base_model_path
        self.lora_ckpt_path = lora_ckpt_path
        self.pooling_method = pooling_method
        self.device = device

        print("🚀 加载Qwen LoRA模型")
        print("=" * 60)

        # 加载tokenizer
        print("📝 加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        if self.pooling_method == "last_token":
            self.tokenizer.padding_side = 'left'

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基模型
        print("🤖 加载基模型...")
        base_model = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )

        # 加载LoRA权重
        print("🔗 加载LoRA权重...")
        self.model = self._load_lora_model(base_model)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print("✅ LoRA模型加载完成")

    def _load_lora_model(self, base_model):
        """加载LoRA模型"""
        safetensors_path = os.path.join(self.lora_ckpt_path, 'adapter_model.safetensors')
        config_path = os.path.join(self.lora_ckpt_path, 'adapter_config.json')

        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"找不到权重文件: {safetensors_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")

        saved_weights = load_file(safetensors_path)
        with open(config_path, 'r') as f:
            config = json.load(f)

        lora_config = LoraConfig(
            r=config.get('r', 8),
            lora_alpha=config.get('lora_alpha', 16),
            target_modules=config.get('target_modules', ['q_proj', 'v_proj', 'k_proj']),
            lora_dropout=0.0,
            bias=config.get('bias', 'none'),
            task_type=config.get('task_type', 'FEATURE_EXTRACTION'),
            init_lora_weights=False
        )

        lora_model = get_peft_model(base_model, lora_config)

        # 加载权重
        loaded_count = self._force_load_lora_weights(lora_model, saved_weights)
        print(f"✅ 成功加载 {loaded_count} 个LoRA权重")

        # 禁用dropout
        self._disable_dropout(lora_model)

        return lora_model

    def _force_load_lora_weights(self, lora_model, saved_weights):
        """强制加载LoRA权重"""
        model_dict = lora_model.state_dict()
        loaded_count = 0

        lora_keys = [k for k in model_dict.keys() if 'lora_' in k]

        for model_key in lora_keys:
            candidates = [
                model_key,
                model_key.replace('base_model.model.', ''),
                'base_model.model.' + model_key,
                model_key.replace('.default.weight', '.weight'),
                model_key.replace('.weight', '.default.weight'),
            ]

            for candidate in candidates:
                if candidate in saved_weights:
                    saved_tensor = saved_weights[candidate]
                    expected_shape = model_dict[model_key].shape

                    if saved_tensor.shape == expected_shape:
                        device = model_dict[model_key].device
                        model_dict[model_key] = saved_tensor.to(device)
                        loaded_count += 1
                        break

        lora_model.load_state_dict(model_dict, strict=False)
        return loaded_count

    def _disable_dropout(self, model):
        """禁用dropout"""
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def last_token_pool(self, last_hidden_states, attention_mask):
        """Last token pooling"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _apply_pooling(self, hidden_states, attention_mask):
        """应用池化"""
        if self.pooling_method == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            return torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_method == "last_token":
            return self.last_token_pool(hidden_states, attention_mask)
        else:
            raise ValueError(f"不支持的pooling方法: {self.pooling_method}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, max_length: int = 8192,
               show_progress: bool = True, normalize: bool = True) -> np.ndarray:
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        pbar = tqdm(range(0, len(texts), batch_size), desc="编码句子") if show_progress else range(0, len(texts), batch_size)

        with torch.no_grad():
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, len(texts))
                batch_sentences = texts[start_idx:end_idx]

                inputs = self.tokenizer(
                    batch_sentences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )

                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

                embeddings = self._apply_pooling(hidden_states, inputs['attention_mask'])

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings


class APIEmbeddingModel(BaseEmbeddingModel):
    """API Embedding模型实现（支持OpenAI、Zhipu等）"""

    def __init__(self, api_provider: str, api_key: str, api_model: str,
                 api_base_url: str = None, max_retries: int = 3, retry_delay: float = 1.0):
        """
        初始化API Embedding模型

        Args:
            api_provider: API提供商 ("openai", "zhipu", "dashscope" 等)
            api_key: API密钥
            api_model: 模型名称
            api_base_url: API基础URL（可选，用于兼容OpenAI接口的服务）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.api_provider = api_provider
        self.api_key = api_key
        self.api_model = api_model
        self.api_base_url = api_base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        print("🚀 初始化API Embedding模型")
        print("=" * 60)
        print(f"📡 API提供商: {api_provider}")
        print(f"🤖 模型: {api_model}")
        if api_base_url:
            print(f"🔗 Base URL: {api_base_url}")

        # 初始化OpenAI客户端（兼容OpenAI接口的服务）
        if api_base_url:
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        print("✅ API客户端初始化完成")

    def _call_api_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        调用API并带重试机制

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.api_model
                )

                # 提取embedding
                embeddings = [data.embedding for data in response.data]
                return embeddings

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠️  API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    print(f"   等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"API调用失败，已重试{self.max_retries}次: {e}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 100,
               show_progress: bool = True, normalize: bool = True) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 单个文本或文本列表
            batch_size: 批次大小（API通常有限制）
            show_progress: 是否显示进度条
            normalize: 是否归一化（API通常已归一化）

        Returns:
            numpy数组，形状为 (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        pbar = tqdm(range(0, len(texts), batch_size), desc="API编码") if show_progress else range(0, len(texts), batch_size)

        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # 调用API
            batch_embeddings = self._call_api_with_retry(batch_texts)
            all_embeddings.extend(batch_embeddings)

            # API限流：短暂延迟
            if end_idx < len(texts):
                time.sleep(0.1)  # 100ms延迟，避免触发限流

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # API返回的embedding通常已经归一化，但可以再次确认
        if normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / np.maximum(norms, 1e-9)

        return embeddings_array


def create_embedding_model(model_type: str, **kwargs) -> BaseEmbeddingModel:
    """
    工厂函数：根据类型创建embedding模型

    Args:
        model_type: 模型类型，"base" | "lora" | "api"
        **kwargs: 模型初始化参数

    Returns:
        BaseEmbeddingModel实例
    """
    if model_type == "base":
        return QwenBaseModel(**kwargs)
    elif model_type == "lora":
        return QwenLoRAModel(**kwargs)
    elif model_type == "api":
        return APIEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
