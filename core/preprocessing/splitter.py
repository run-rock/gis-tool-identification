"""
阶段1：数据合并、打乱、分割

将多个工具的训练样本文件合并，打乱顺序，按比例划分为训练集和验证集。
"""

import json
import os
import random
from typing import Dict, Any

from .config import PreprocessConfig


class DataSplitter:
    """
    数据合并、打乱、分割处理器

    功能：
    1. 收集所有 *_training_samples.json 文件
    2. 合并所有案例
    3. 打乱数据（使用随机种子）
    4. 按比例划分训练集和验证集
    5. 保存为 train_samples.json 和 val_samples.json

    注意：完全复制原 merge_shuffle_split.py 的逻辑，保证输出一致性
    """

    def __init__(self, config: PreprocessConfig):
        """
        初始化数据分割器

        Args:
            config: 预处理配置
        """
        self.config = config

    def run(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        运行数据合并、打乱、分割

        Args:
            input_dir: 输入目录（包含多个 *_training_samples.json 文件）
            output_dir: 输出目录

        Returns:
            统计信息字典，包含：
            - total_files: 找到的样本文件数
            - total_cases: 总案例数
            - train_count: 训练集案例数
            - val_count: 验证集案例数
            - train_file: 训练集文件路径
            - val_file: 验证集文件路径
        """
        print(f"🔄 开始合并、打乱和划分训练样本")
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 训练集:验证集 = {self.config.train_ratio}:{1-self.config.train_ratio}")
        print(f"🎲 随机种子: {self.config.random_seed}")
        print()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 查找所有训练样本文件
        sample_files = []
        for file in os.listdir(input_dir):
            if file.endswith('_training_samples.json'):
                sample_files.append(os.path.join(input_dir, file))

        print(f"📄 找到 {len(sample_files)} 个训练样本文件\n")

        # 收集所有案例
        all_cases = []

        # 注意：使用 sorted 保证文件处理顺序一致
        for file_path in sorted(sample_files):
            tool_name = os.path.basename(file_path).replace('_training_samples.json', '')

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取每个样本
                for sample in data['samples']:
                    all_cases.append(sample)

                print(f"✅ {tool_name}: {len(data['samples'])} 个案例")

            except Exception as e:
                print(f"❌ {tool_name}: 读取失败 - {e}")
                continue

        print(f"\n📊 总共收集了 {len(all_cases)} 个案例")

        # 打乱数据（注意：随机种子设置时机必须与原代码一致）
        print(f"\n🔀 正在打乱数据...")
        random.seed(self.config.random_seed)
        random.shuffle(all_cases)
        print(f"✅ 数据已打乱（随机种子={self.config.random_seed}）")

        # 划分训练集和验证集
        total_count = len(all_cases)
        train_count = int(total_count * self.config.train_ratio)

        train_data = all_cases[:train_count]
        val_data = all_cases[train_count:]

        print(f"\n📊 划分结果:")
        print(f"   训练集: {len(train_data)} 个案例 ({len(train_data)/total_count:.2%})")
        print(f"   验证集: {len(val_data)} 个案例 ({len(val_data)/total_count:.2%})")

        # 保存训练集
        train_file = os.path.join(output_dir, "train_samples.json")
        print(f"\n💾 保存训练集: {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        # 保存验证集
        val_file = os.path.join(output_dir, "val_samples.json")
        print(f"💾 保存验证集: {val_file}")
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 完成!")
        print(f"📂 输出目录: {output_dir}")
        print(f"   - train_samples.json: {len(train_data)} 个案例")
        print(f"   - val_samples.json: {len(val_data)} 个案例")

        return {
            'total_files': len(sample_files),
            'total_cases': total_count,
            'train_count': len(train_data),
            'val_count': len(val_data),
            'train_file': train_file,
            'val_file': val_file
        }
