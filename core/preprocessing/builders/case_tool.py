"""
案例-工具样本构建器

对于每个案例：
- 遍历所有正确工具（correct_tools）
- 对于每个正确工具，收集所有错误工具（incorrect_tools）和随机负样本到一个列表
- 生成格式：{案例描述, 正确工具, 错误工具列表}

示例：
案例A有3个正确工具、1个错误工具 → 生成 3 个三元组，每个包含1+2=3个负样本
案例B有4个正确工具、0个错误工具 → 生成 4 个三元组，每个包含2个负样本
"""

import json
import random
from typing import Dict, Any, List
from tqdm import tqdm

from .base import BaseSampleBuilder
from ..formatters import (
    format_tool_info,
    extract_algorithm_ids,
    load_all_tools
)


class CaseToolBuilder(BaseSampleBuilder):
    """案例→工具样本构建器"""

    def build_samples(
        self,
        input_file: str,
        output_file: str,
        tools_file: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建案例-工具三元组数据

        完全复制原 build_case_tool_sample.py 的逻辑

        Args:
            input_file: 输入JSON文件路径（train_samples.json 或 val_samples.json）
            output_file: 输出JSONL文件路径
            tools_file: 工具数据文件路径（如果为None，使用config中的默认值）

        Returns:
            统计信息
        """
        if tools_file is None:
            tools_file = self.config.tools_file

        print(f"🔄 开始构建案例-工具三元组数据")
        print(f"📁 输入文件: {input_file}")
        print(f"📁 输出文件: {output_file}")
        print(f"📦 工具文件: {tools_file}")
        print(f"🎲 每个正样本配对的随机负样本数: {self.config.random_negatives_per_positive}")
        print(f"🎲 随机种子: {self.config.random_seed}")
        print()

        # 设置随机种子（时机与原代码一致）
        random.seed(self.config.random_seed)

        # 加载所有工具
        print(f"📂 加载工具库...")
        all_tools = load_all_tools(tools_file)
        print()

        # 读取数据
        print(f"📂 读取输入文件...")
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        print(f"✅ 读取了 {len(samples)} 个案例")

        # 构建三元组
        print(f"\n🔄 构建三元组...")
        triplets = []
        skipped_cases = 0

        for sample in tqdm(samples, desc="处理案例"):
            case_description = sample['case_description']
            correct_tools = sample.get('correct_tools', [])
            incorrect_tools = sample.get('incorrect_tools', [])

            # 如果没有正确工具，跳过
            if not correct_tools:
                skipped_cases += 1
                continue

            # 提取当前案例中已存在的工具的algorithm_id
            existing_ids = extract_algorithm_ids(correct_tools + incorrect_tools)

            # 遍历所有正确工具
            for correct_tool in correct_tools:
                # 收集负样本列表
                negative_tools = []

                # 1. 先添加所有错误工具作为负样本
                negative_tools.extend(incorrect_tools)

                # 2. 为每个正确工具添加随机负样本
                # 过滤出可用的工具（不在已有工具中）
                available_tools = [
                    tool for tool in all_tools
                    if tool.get('algorithm_id', '') not in existing_ids
                ]

                # 随机选择指定数量的工具
                if len(available_tools) >= self.config.random_negatives_per_positive:
                    sampled_tools = random.sample(available_tools, self.config.random_negatives_per_positive)
                else:
                    sampled_tools = available_tools  # 如果可用工具不足，全部使用

                # 将随机工具格式化后添加到负样本列表
                for random_tool in sampled_tools:
                    random_tool_text = format_tool_info(random_tool)
                    negative_tools.append(random_tool_text)

                # 创建三元组：一个正样本对应多个负样本
                triplet = {
                    'case_description': case_description,
                    'correct_tool': correct_tool,
                    'incorrect_tools': negative_tools
                }
                triplets.append(triplet)

                # 更新已使用的工具ID，避免下一个正样本重复抽取
                for tool in sampled_tools:
                    existing_ids.add(tool.get('algorithm_id', ''))

        print(f"✅ 构建了 {len(triplets)} 个三元组")
        if skipped_cases > 0:
            print(f"⚠️  跳过了 {skipped_cases} 个没有正确工具的案例")

        # 统计信息
        total_cases = len(samples) - skipped_cases
        avg_triplets_per_case = len(triplets) / total_cases if total_cases > 0 else 0

        print(f"\n📊 统计信息:")
        print(f"   有效案例数: {total_cases}")
        print(f"   总三元组数: {len(triplets)}")
        print(f"   平均每个案例的三元组数: {avg_triplets_per_case:.2f}")

        # 保存为JSONL格式
        print(f"\n💾 保存为JSONL格式: {output_file}")
        self._save_jsonl(triplets, output_file)

        print(f"✅ 完成! 共 {len(triplets)} 个三元组")

        return {
            'total_triplets': len(triplets),
            'total_cases': len(samples),
            'valid_cases': total_cases,
            'skipped_cases': skipped_cases,
            'avg_triplets_per_case': avg_triplets_per_case
        }
