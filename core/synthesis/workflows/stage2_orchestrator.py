#!/usr/bin/env python3
"""
第二阶段编排器

协调相似工具搜索和评估的完整工作流
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..services.vector_db import VectorDatabase
from ..services.llm_utils import evaluate_similar_tool_stage1, evaluate_similar_tool_stage2
from ..prompts import JudgeResult


class Stage2Orchestrator:
    """
    第二阶段编排器

    负责：
    1. 为案例搜索相似工具（使用向量数据库）
    2. 两阶段评估相似工具
    3. 生成最终结果
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        llm,
        top_k: int = 3,
        max_retries: int = 3
    ):
        """
        初始化第二阶段编排器

        Args:
            vector_db: 向量数据库实例（已加载）
            llm: LLM实例（用于相似工具评估）
            top_k: 每个案例搜索的相似工具数量
            max_retries: LLM调用最大重试次数
        """
        self.vector_db = vector_db
        self.llm = llm
        self.top_k = top_k
        self.max_retries = max_retries

    def _get_judge_text(self, judge: int) -> str:
        """获取判断结果的文本描述"""
        if judge == JudgeResult.CAN_SOLVE:
            return "✅ 能独立解决 (judge=1)"
        elif judge == JudgeResult.CANNOT_SOLVE:
            return "❌ 不能独立解决 (judge=0)"
        elif judge == JudgeResult.REQUIRES_COMBINATION:
            return "⚠️ 需要辅助 (judge=2)"
        else:
            return "❌ 检查失败"

    def evaluate_single_tool(
        self,
        case_text: str,
        tool: Dict
    ) -> Dict:
        """
        两阶段评估单个相似工具

        Args:
            case_text: 案例文本
            tool: 工具信息（包含name, algorithm_id, description, parameters）

        Returns:
            评估结果字典
        """
        # 阶段1：检查数据和功能匹配性
        print(f"      🔍 阶段1：检查数据和功能匹配性")
        stage1_result = evaluate_similar_tool_stage1(
            case_text,
            tool,
            self.llm,
            self.max_retries
        )

        judge_text_1 = self._get_judge_text(stage1_result['judge'])
        print(f"      {judge_text_1}")
        print(f"      理由: {stage1_result['evidence']}")

        # 如果第一阶段不是1，直接返回结果
        if stage1_result['judge'] != JudgeResult.CAN_SOLVE:
            return {
                "tool": tool,
                "stage1": stage1_result,
                "stage2": None,
                "final_judge": stage1_result['judge'],
                "final_judge_text": judge_text_1,
                "final_evidence": f"阶段1判定: {stage1_result['evidence']}",
                "is_valid": False
            }

        # 阶段2：检查输出是否满足需求
        print(f"      🔍 阶段2：检查输出是否满足需求")
        stage2_result = evaluate_similar_tool_stage2(
            case_text,
            tool,
            self.llm,
            self.max_retries
        )

        judge_text_2 = self._get_judge_text(stage2_result['judge'])
        print(f"      {judge_text_2}")
        print(f"      理由: {stage2_result['evidence']}")

        # 两阶段都通过才算有效
        is_valid = (
            stage1_result['judge'] == JudgeResult.CAN_SOLVE and
            stage2_result['judge'] == JudgeResult.CAN_SOLVE
        )

        # 最终判定取两阶段中较低的值（更保守）
        final_judge = min(stage1_result['judge'], stage2_result['judge'])
        final_judge_text = self._get_judge_text(final_judge)

        return {
            "tool": tool,
            "stage1": stage1_result,
            "stage2": stage2_result,
            "final_judge": final_judge,
            "final_judge_text": final_judge_text,
            "final_evidence": f"阶段1: {stage1_result['evidence']}\n阶段2: {stage2_result['evidence']}",
            "is_valid": is_valid
        }

    def evaluate_all_tools(
        self,
        case_text: str,
        similar_tools: List[Dict]
    ) -> List[Dict]:
        """
        评估所有相似工具

        Args:
            case_text: 案例文本
            similar_tools: 相似工具列表

        Returns:
            评估结果列表
        """
        all_tool_results = []

        for i, tool in enumerate(similar_tools, 1):
            print(f"    检查工具 {i}/{len(similar_tools)}: {tool['name']}")

            # 评估单个工具（两阶段）
            tool_result = self.evaluate_single_tool(case_text, tool)
            all_tool_results.append(tool_result)

            # 显示最终检查结果
            print(f"      ⭐ 最终结果: {tool_result['final_judge_text']}")
            print()

        return all_tool_results

    def process_single_case(
        self,
        case: Dict,
        current_tool_id: str
    ) -> Dict:
        """
        处理单个案例：搜索相似工具 + 评估

        Args:
            case: 案例字典（包含generated_case等字段）
            current_tool_id: 当前工具的algorithm_id（用于排除）

        Returns:
            处理后的案例（添加了similar_tools_check字段）
        """
        case_text = case['generated_case']

        # 搜索相似工具
        similar_tools = self.vector_db.search_similar_tools(
            query=case_text,
            k=self.top_k,
            exclude_tool_id=current_tool_id
        )

        print(f"    📦 找到 {len(similar_tools)} 个相似工具")

        # 评估所有相似工具
        print(f"    🔍 开始逐个评估相似工具：")
        all_tool_results = self.evaluate_all_tools(case_text, similar_tools)

        # 统计有效工具
        valid_tools = []
        for result in all_tool_results:
            if result['is_valid']:
                tool_info = {
                    "name": result['tool']['name'],
                    "description": result['tool']['description'],
                    "algorithm_id": result['tool']['algorithm_id'],
                    "parameters": result['tool'].get('parameters', {}),
                    "stage1_evidence": result['stage1']['evidence'],
                    "stage2_evidence": result['stage2']['evidence'],
                    "final_evidence": result['final_evidence'],
                    "check_timestamp": datetime.now().isoformat()
                }
                valid_tools.append(tool_info)

        # 构建处理后的案例（保留原有信息）
        processed_case = case.copy()

        # 添加相似工具列表（简化版，用于训练数据）
        processed_case['similar_tools'] = [
            {
                "name": tool['name'],
                "algorithm_id": tool['algorithm_id'],
                "description": tool['description'],
                "parameters": tool.get('parameters', {})
            }
            for tool in similar_tools
        ]

        # 添加相似工具检查结果
        processed_case['similar_tools_check'] = {
            "total_similar_tools_checked": len(similar_tools),
            "valid_similar_tools_count": len(valid_tools),
            "all_tool_check_results": all_tool_results,  # 保存所有工具的检查结果
            "valid_similar_tools": valid_tools
        }

        print(f"    🎯 总结: 找到 {len(valid_tools)}/{len(all_tool_results)} 个可用的相似工具")

        return processed_case

    def run(
        self,
        cases_file: str,
        output_file: str
    ) -> Dict:
        """
        运行完整的第二阶段处理

        Args:
            cases_file: 第一阶段输出的案例文件（final_cases_*.json）
            output_file: 输出文件路径

        Returns:
            处理报告
        """
        print(f"🚀 开始第二阶段处理: {os.path.basename(cases_file)}")

        # 读取案例文件
        with open(cases_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tool_info = data['tool']
        toolbox_info = data['toolbox']
        cases = data['cases']

        print(f"📋 找到 {len(cases)} 个案例")
        print(f"🔧 原工具: {tool_info['name']}")

        # 获取当前工具的algorithm_id用于排除
        current_tool_id = tool_info.get('algorithm_id', '')

        # 处理每个案例
        processed_cases = []
        total_added_tools = 0

        for i, case in enumerate(cases, 1):
            print(f"\n" + "="*80)
            print(f"🔍 处理案例 {i}/{len(cases)}")
            print(f"="*80)

            # 显示案例详情
            case_text = case['generated_case']
            print(f"📋 案例详情:")
            print(f"  场景: {case.get('scenario', {}).get('name', '未知场景')}")
            print(f"  案例内容: {case_text[:100]}..." if len(case_text) > 100 else f"  案例内容: {case_text}")
            print()

            # 处理单个案例
            processed_case = self.process_single_case(case, current_tool_id)
            processed_cases.append(processed_case)

            total_added_tools += processed_case['similar_tools_check']['valid_similar_tools_count']

        # 构建最终结果（保留原有结构）
        final_result = data.copy()
        final_result['cases'] = processed_cases

        # 添加处理信息
        final_result['similar_tools_processing'] = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_cases_processed": len(processed_cases),
            "total_similar_tools_added": total_added_tools,
            "vector_db_size": len(self.vector_db)
        }

        # 保存结果文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        print(f"\n📄 结果已保存到: {output_file}")

        # 生成报告
        report = {
            "summary": {
                "total_cases": len(processed_cases),
                "total_similar_tools_added": total_added_tools,
                "processing_success": True
            },
            "input_file": cases_file,
            "output_file": output_file
        }

        return report
