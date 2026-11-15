"""
工具-工具箱样本构建器

对于每个工具：
- 正样本：该工具所属的工具箱
- 负样本：从其他工具箱中随机选择
- 生成格式：{工具, 正确工具箱, 错误工具箱列表}
"""

import json
import random
from typing import Dict, Any, List
from tqdm import tqdm

from .base import BaseSampleBuilder
from ..formatters import format_tool_info


class ToolToolboxBuilder(BaseSampleBuilder):
    """工具→工具箱样本构建器"""

    def build_samples(
        self,
        input_file: str,
        output_file: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建工具-工具箱三元组数据

        完全复制原 build_tool_toolbox_samples.py 的逻辑

        Args:
            input_file: 输入JSON文件路径（qgis_tool_with_paramaters重写工具集描述.json）
            output_file: 输出JSONL文件路径

        Returns:
            统计信息
        """
        negative_count = self.config.tool_toolbox_negatives

        print(f"🔄 开始构建工具-工具箱训练样本")
        print(f"📁 输入文件: {input_file}")
        print(f"📁 输出文件: {output_file}")
        print(f"🎯 负样本工具箱数量: {negative_count}")
        print(f"🎲 随机种子: {self.config.random_seed}")
        print()

        # 设置随机种子
        random.seed(self.config.random_seed)

        # 读取数据
        print(f"📂 读取输入文件...")
        with open(input_file, 'r', encoding='utf-8') as f:
            toolboxes_data = json.load(f)

        print(f"✅ 读取了 {len(toolboxes_data)} 个工具箱")

        # 收集所有工具箱信息
        all_toolboxes = []
        for toolbox in toolboxes_data:
            toolbox_info = {
                'toolbox_name': toolbox.get('toolbox', ''),
                'toolbox_description': toolbox.get('discription', '')  # 注意：原文件使用 'discription'
            }
            all_toolboxes.append(toolbox_info)

        print(f"✅ 收集了 {len(all_toolboxes)} 个工具箱")

        # 为每个工具构建训练样本
        print(f"\n🔄 为每个工具构建训练样本...")
        tool_toolbox_samples = []
        total_tools = 0

        for toolbox in tqdm(toolboxes_data, desc="处理工具箱"):
            toolbox_name = toolbox.get('toolbox', '')
            toolbox_desc = toolbox.get('discription', '')
            tools = toolbox.get('tools', [])

            # 当前工具箱信息（正样本）
            correct_toolbox = {
                'toolbox_name': toolbox_name,
                'toolbox_description': toolbox_desc
            }

            # 候选负样本：其他工具箱
            candidate_toolboxes = [
                tb for tb in all_toolboxes
                if tb['toolbox_name'] != toolbox_name
            ]

            # 为该工具箱中的每个工具创建样本
            for tool in tools:
                algorithm_id = tool.get('algorithm_id', '')
                if not algorithm_id:
                    continue

                # 格式化工具信息
                tool_text = format_tool_info(tool)

                # 随机选择负样本工具箱
                if len(candidate_toolboxes) >= negative_count:
                    incorrect_toolboxes = random.sample(candidate_toolboxes, negative_count)
                else:
                    incorrect_toolboxes = candidate_toolboxes  # 如果候选不足，全部使用

                # 构建训练样本
                tool_sample = {
                    'tool': tool_text,
                    'correct_toolbox': correct_toolbox,
                    'incorrect_toolboxes': incorrect_toolboxes
                }

                tool_toolbox_samples.append(tool_sample)
                total_tools += 1

        print(f"✅ 构建了 {len(tool_toolbox_samples)} 个工具训练样本")

        # 统计信息
        total_incorrect_toolboxes = sum(len(s['incorrect_toolboxes']) for s in tool_toolbox_samples)
        avg_incorrect = total_incorrect_toolboxes / len(tool_toolbox_samples) if tool_toolbox_samples else 0

        print(f"\n📊 统计信息:")
        print(f"   总工具数: {total_tools}")
        print(f"   平均负样本工具箱数: {avg_incorrect:.2f}")
        print(f"   总负样本工具箱数: {total_incorrect_toolboxes}")

        # 保存为JSONL格式
        print(f"\n💾 保存为JSONL格式: {output_file}")
        self._save_jsonl(tool_toolbox_samples, output_file)

        print(f"✅ 完成! 共 {len(tool_toolbox_samples)} 个工具样本")

        return {
            'total_samples': len(tool_toolbox_samples),
            'total_tools': total_tools,
            'avg_incorrect_toolboxes': avg_incorrect,
            'total_incorrect_toolboxes': total_incorrect_toolboxes
        }
