"""
工具箱-案例样本构建器（反向）

对于每个工具箱：
- 从训练样本中找到所有包含该工具箱的案例作为正样本
- 随机选择不包含该工具箱的案例作为负样本
- 生成格式：{工具箱, 正确案例, 错误案例列表}
"""

import json
import random
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm

from .base import BaseSampleBuilder
from ..formatters import extract_toolbox_name


class ToolboxCaseBuilder(BaseSampleBuilder):
    """工具箱→案例样本构建器"""

    def build_samples(
        self,
        input_file: str,
        output_file: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建工具箱-案例三元组数据（反向）

        完全复制原 build_revers_tool_toolbox_samples.py 的 toolbox 模式逻辑

        Args:
            input_file: 输入JSON文件路径（train_samples.json 或 val_samples.json）
            output_file: 输出JSONL文件路径

        Returns:
            统计信息
        """
        print(f"🔄 开始构建工具箱-案例三元组数据（反向）")
        print(f"📁 输入文件: {input_file}")
        print(f"📁 输出文件: {output_file}")
        print(f"🎲 每个正样本配对的随机负样本数: {self.config.random_negatives_per_positive}")
        print(f"🎲 随机种子: {self.config.random_seed}")
        print()

        # 设置随机种子（时机与原代码一致）
        random.seed(self.config.random_seed)

        # 读取数据
        print(f"📂 读取输入文件...")
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        print(f"✅ 读取了 {len(samples)} 个案例")

        # 第一步：建立工具箱到案例的映射
        print(f"\n🔄 建立工具箱到案例的映射...")
        toolbox_to_cases: Dict[str, List[str]] = defaultdict(list)
        toolbox_info_map: Dict[str, Dict] = {}  # toolbox_name -> 工具箱信息

        for sample in tqdm(samples, desc="建立映射"):
            case_description = sample['case_description']
            correct_toolboxes = sample.get('correct_toolboxes', [])

            # 对于每个正确工具箱，记录它对应的案例
            for toolbox in correct_toolboxes:
                if isinstance(toolbox, dict):
                    toolbox_name = extract_toolbox_name(toolbox)
                    if toolbox_name:
                        toolbox_to_cases[toolbox_name].append(case_description)
                        # 保存工具箱信息（用于输出）
                        if toolbox_name not in toolbox_info_map:
                            toolbox_info_map[toolbox_name] = toolbox

        print(f"✅ 找到 {len(toolbox_to_cases)} 个有正样本的工具箱")

        # 收集所有案例描述（用于负采样）
        all_case_descriptions = [sample['case_description'] for sample in samples]

        # 第二步：为每个工具箱的每个正样本案例构建三元组
        print(f"\n🔄 构建三元组...")
        triplets = []

        for toolbox_name, positive_cases in tqdm(toolbox_to_cases.items(), desc="处理工具箱"):
            toolbox_info = toolbox_info_map[toolbox_name]

            # 找出不包含该工具箱的案例（候选负样本）
            negative_candidates = [
                case for case in all_case_descriptions
                if case not in positive_cases
            ]

            # 为该工具箱的每个正样本案例创建一个三元组
            for positive_case in positive_cases:
                # 随机选择负样本案例
                if len(negative_candidates) >= self.config.random_negatives_per_positive:
                    negative_cases = random.sample(negative_candidates, self.config.random_negatives_per_positive)
                else:
                    negative_cases = negative_candidates  # 如果候选不足，全部使用

                # 创建三元组
                triplet = {
                    'toolbox': toolbox_info,
                    'correct_case': positive_case,
                    'incorrect_cases': negative_cases
                }
                triplets.append(triplet)

        print(f"✅ 构建了 {len(triplets)} 个三元组")

        # 统计信息
        total_toolboxes = len(toolbox_to_cases)
        avg_triplets_per_toolbox = len(triplets) / total_toolboxes if total_toolboxes > 0 else 0

        print(f"\n📊 统计信息:")
        print(f"   总工具箱数: {total_toolboxes}")
        print(f"   总三元组数: {len(triplets)}")
        print(f"   平均每个工具箱的三元组数: {avg_triplets_per_toolbox:.2f}")

        # 保存为JSONL格式
        print(f"\n💾 保存为JSONL格式: {output_file}")
        self._save_jsonl(triplets, output_file)

        print(f"✅ 完成! 共 {len(triplets)} 个三元组")

        return {
            'total_triplets': len(triplets),
            'total_toolboxes': total_toolboxes,
            'avg_triplets_per_toolbox': avg_triplets_per_toolbox
        }
