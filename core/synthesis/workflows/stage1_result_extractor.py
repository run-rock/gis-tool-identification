#!/usr/bin/env python3
"""
第一阶段结果提取器

从工作流结果中提取最终有效案例，为第二阶段做准备
"""

import os
import json
from typing import Dict, Optional
from datetime import datetime


class Stage1ResultExtractor:
    """
    第一阶段结果提取器

    负责从第一阶段的工作流结果中提取最终有效案例：
    1. 从初始检查中提取通过的案例 (judge=1)
    2. 从最终检查中提取修正后通过的案例 (judge=1)
    3. 合并并保存为统一格式
    """

    def __init__(self):
        """初始化提取器"""
        pass

    def extract_from_directory(
        self,
        tool_workflow_dir: str,
        output_dir: str
    ) -> Dict:
        """
        从工具工作流目录中提取最终案例

        Args:
            tool_workflow_dir: 工具工作流目录（包含所有工作流文件）
            output_dir: 输出目录

        Returns:
            提取报告
        """
        print(f"🔄 开始提取案例最终状态")
        print(f"📁 工具工作流目录: {tool_workflow_dir}")

        # 查找工作流文件
        check_result_files = [
            f for f in os.listdir(tool_workflow_dir)
            if f.startswith('check_result_') and f.endswith('.json')
        ]
        final_check_files = [
            f for f in os.listdir(tool_workflow_dir)
            if f.startswith('final_check_') and f.endswith('.json')
        ]

        if not check_result_files:
            raise ValueError(
                f"在目录 {tool_workflow_dir} 中找不到初始检查文件 (check_result_*.json)"
            )

        # 读取文件
        initial_check_file = os.path.join(tool_workflow_dir, check_result_files[0])
        final_check_file = (
            os.path.join(tool_workflow_dir, final_check_files[0])
            if final_check_files else None
        )

        print(f"📄 初始检查文件: {initial_check_file}")
        print(f"📄 最终检查文件: {final_check_file if final_check_file else '无'}")

        # 提取案例
        final_data = self._extract_final_cases(initial_check_file, final_check_file)

        # 生成输出文件名
        tool_id = final_data['tool']['algorithm_id'].replace(':', '_')
        output_filename = f"final_cases_{tool_id}.json"
        output_file = os.path.join(output_dir, output_filename)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        # 打印统计信息
        extraction_info = final_data['extraction_info']
        total_cases = extraction_info['total_final_cases']
        initial_count = extraction_info['initial_valid_cases']
        final_count = extraction_info['final_valid_cases']

        print(f"\n✅ 案例最终状态提取完成")
        print(f"📊 统计信息:")
        print(f"  总案例数: {total_cases}")
        print(f"  初始通过: {initial_count} 个")
        print(f"  修正通过: {final_count} 个")
        print(f"📁 输出文件: {output_file}")

        return {
            'output_file': output_file,
            'total_cases': total_cases,
            'initial_valid_cases': initial_count,
            'final_valid_cases': final_count
        }

    def _load_json_file(self, file_path: str) -> Optional[Dict]:
        """
        安全地加载JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            JSON数据字典，如果文件不存在或路径为None返回None
        """
        if file_path is None or not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")
            return None

    def _extract_final_cases(
        self,
        initial_check_file: str,
        final_check_file: Optional[str]
    ) -> Dict:
        """
        提取案例的最终状态

        Args:
            initial_check_file: 初始检查结果文件路径
            final_check_file: 最终检查结果文件路径（修正后的检查结果）

        Returns:
            包含最终案例状态的字典
        """
        # 读取初始检查结果
        initial_data = self._load_json_file(initial_check_file)
        if not initial_data:
            raise ValueError(f"无法读取初始检查文件: {initial_check_file}")

        # 读取最终检查结果（如果存在）
        final_data = self._load_json_file(final_check_file)

        # 提取工具和工具箱信息
        tool_info = initial_data['tool_info']
        toolbox_info = initial_data['toolbox_info']

        final_cases = []

        # 1. 从初始检查中提取所有通过的案例 (is_valid: true)
        initial_valid_count = 0
        for case_item in initial_data['checked_cases']:
            if case_item['check_result']['is_valid']:
                case_data = case_item['original_case']
                check_result = case_item['check_result']

                final_case = {
                    "generated_case": case_data['generated_case'],
                    "scenario": case_data['scenario'],
                    "check_result": {
                        "judge": check_result['judge'],
                        "evidence": check_result['evidence'],
                        "is_valid": check_result['is_valid'],
                        "check_type": "initial",  # 标记这是初始检查通过的案例
                        "check_timestamp": case_item['check_timestamp']
                    }
                }
                final_cases.append(final_case)
                initial_valid_count += 1

        print(f"📋 从初始检查中提取了 {initial_valid_count} 个通过的案例")

        # 2. 从最终检查中提取通过的案例（judge=1）
        final_valid_count = 0
        if final_data is not None:
            for case_item in final_data['checked_cases']:
                if case_item['check_result']['is_valid']:  # 只保存通过的案例
                    case_data = case_item['original_case']
                    check_result = case_item['check_result']

                    final_case = {
                        "generated_case": case_data['generated_case'],
                        "scenario": case_data['scenario'],
                        "check_result": {
                            "judge": check_result['judge'],
                            "evidence": check_result['evidence'],
                            "is_valid": check_result['is_valid'],
                            "check_type": "final",  # 标记这是最终检查的案例（修正后）
                            "check_timestamp": case_item['check_timestamp']
                        }
                    }
                    final_cases.append(final_case)
                    final_valid_count += 1

            print(f"📋 从最终检查中提取了 {final_valid_count} 个通过的案例（修正后通过）")
        else:
            print("📋 没有最终检查文件，所有案例都在初始检查中通过")

        # 构建最终结果
        result = {
            "tool": tool_info,
            "toolbox": toolbox_info,
            "extraction_info": {
                "initial_check_file": initial_check_file,
                "final_check_file": final_check_file if final_data else None,
                "extraction_timestamp": datetime.now().isoformat(),
                "total_final_cases": len(final_cases),
                "initial_valid_cases": initial_valid_count,
                "final_valid_cases": final_valid_count
            },
            "cases": final_cases
        }

        return result
