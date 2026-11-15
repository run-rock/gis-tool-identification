"""
工具箱-工具样本构建器

对于每个工具箱的每个工具：
- 正样本：该工具箱中的某个工具
- 负样本：从其他工具箱的工具中随机选择
- 生成格式：{工具箱, 正确工具, 错误工具列表}
"""

import json
import random
from typing import Dict, Any, List
from tqdm import tqdm

from .base import BaseSampleBuilder
from ..formatters import format_tool_info


class ToolboxToolBuilder(BaseSampleBuilder):
    """工具箱→工具样本构建器"""

    def build_samples(
        self,
        input_file: str,
        output_file: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建工具箱-工具三元组数据

        完全复制原 build_toolbox_tool_samples.py 的逻辑

        Args:
            input_file: 输入JSON文件路径（qgis_tool_with_paramaters重写工具集描述.json）
            output_file: 输出JSONL文件路径

        Returns:
            统计信息
        """
        negative_count = self.config.toolbox_tool_negatives

        print(f"🔄 开始构建工具箱-工具训练样本")
        print(f"📁 输入文件: {input_file}")
        print(f"📁 输出文件: {output_file}")
        print(f"🎯 负样本工具数量: {negative_count}")
        print(f"🎲 随机种子: {self.config.random_seed}")
        print()

        # 设置随机种子
        random.seed(self.config.random_seed)

        # 读取数据
        print(f"📂 读取输入文件...")
        with open(input_file, 'r', encoding='utf-8') as f:
            toolboxes_data = json.load(f)

        print(f"✅ 读取了 {len(toolboxes_data)} 个工具箱")

        # 收集所有工具（用于负采样）
        print(f"\n🔄 收集所有工具...")
        all_tools = []  # 每个元素是 (toolbox_name, tool_text)
        toolbox_tool_map: Dict[str, List[str]] = {}  # toolbox_name -> [tool_text, ...]

        for toolbox in toolboxes_data:
            toolbox_name = toolbox.get('toolbox', '')
            tools = toolbox.get('tools', [])
            toolbox_tool_map[toolbox_name] = []

            for tool in tools:
                tool_text = format_tool_info(tool)
                all_tools.append((toolbox_name, tool_text))
                toolbox_tool_map[toolbox_name].append(tool_text)

        print(f"✅ 收集了 {len(all_tools)} 个工具")

        # 为每个工具箱的每个工具构建训练样本
        print(f"\n🔄 为每个工具箱的每个工具构建训练样本...")
        toolbox_tool_samples = []

        for toolbox in tqdm(toolboxes_data, desc="处理工具箱"):
            toolbox_name = toolbox.get('toolbox', '')
            toolbox_desc = toolbox.get('discription', '')
            tools = toolbox.get('tools', [])

            # 当前工具箱信息
            toolbox_info = {
                'toolbox_name': toolbox_name,
                'toolbox_description': toolbox_desc
            }

            # 候选负样本：其他工具箱的工具
            candidate_tools = [
                tool_text for tb_name, tool_text in all_tools
                if tb_name != toolbox_name
            ]

            # 为该工具箱中的每个工具创建样本
            for tool in tools:
                algorithm_id = tool.get('algorithm_id', '')
                if not algorithm_id:
                    continue

                # 格式化工具信息（正样本）
                correct_tool = format_tool_info(tool)

                # 随机选择负样本工具
                if len(candidate_tools) >= negative_count:
                    incorrect_tools = random.sample(candidate_tools, negative_count)
                else:
                    incorrect_tools = candidate_tools  # 如果候选不足，全部使用

                # 构建训练样本
                toolbox_sample = {
                    'toolbox': toolbox_info,
                    'correct_tool': correct_tool,
                    'incorrect_tools': incorrect_tools
                }

                toolbox_tool_samples.append(toolbox_sample)

        print(f"✅ 构建了 {len(toolbox_tool_samples)} 个工具箱训练样本")

        # 统计信息
        total_incorrect_tools = sum(len(s['incorrect_tools']) for s in toolbox_tool_samples)
        avg_incorrect = total_incorrect_tools / len(toolbox_tool_samples) if toolbox_tool_samples else 0

        print(f"\n📊 统计信息:")
        print(f"   总样本数: {len(toolbox_tool_samples)}")
        print(f"   平均负样本工具数: {avg_incorrect:.2f}")
        print(f"   总负样本工具数: {total_incorrect_tools}")

        # 保存为JSONL格式
        print(f"\n💾 保存为JSONL格式: {output_file}")
        self._save_jsonl(toolbox_tool_samples, output_file)

        print(f"✅ 完成! 共 {len(toolbox_tool_samples)} 个工具箱样本")

        return {
            'total_samples': len(toolbox_tool_samples),
            'avg_incorrect_tools': avg_incorrect,
            'total_incorrect_tools': total_incorrect_tools
        }
