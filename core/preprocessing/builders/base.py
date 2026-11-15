"""
样本构建器抽象基类
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from ..config import PreprocessConfig


class BaseSampleBuilder(ABC):
    """
    样本构建器抽象基类

    定义样本构建的通用接口，但不强制实现细节。
    每个子类可以完全按照原代码逻辑实现，保证输出一致性。
    """

    def __init__(self, config: PreprocessConfig):
        """
        初始化样本构建器

        Args:
            config: 预处理配置
        """
        self.config = config

    @abstractmethod
    def build_samples(self, input_file: str, output_file: str, **kwargs) -> Dict[str, Any]:
        """
        构建样本（子类必须实现）

        Args:
            input_file: 输入文件路径
            output_file: 输出JSONL文件路径
            **kwargs: 其他参数（如tools_file）

        Returns:
            统计信息字典
        """
        pass

    def _save_jsonl(self, triplets: List[Dict], output_file: str):
        """
        保存为JSONL格式（通用逻辑）

        Args:
            triplets: 三元组列表
            output_file: 输出文件路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for triplet in triplets:
                json_line = json.dumps(triplet, ensure_ascii=False)
                f.write(json_line + '\n')
