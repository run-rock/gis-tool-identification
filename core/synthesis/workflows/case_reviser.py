#!/usr/bin/env python3
"""
案例修正器 - 修正无效案例
"""

from typing import List
from datetime import datetime

from ..base import BaseWorkflow
from ..models import Case, CaseStatus, Tool
from ..services.llm_utils import revise_case_with_llm


class CaseReviser(BaseWorkflow):
    """
    案例修正器

    职责：根据验证反馈修正无效案例
    """

    def __init__(self, llm, output_dir: str = None):
        """
        初始化案例修正器

        Args:
            llm: LLM模型实例（建议temperature=0.7，更有创造性）
            output_dir: 输出目录
        """
        super().__init__(llm, output_dir)

    def execute(self, invalid_cases: List[Case], tool: Tool) -> List[Case]:
        """
        修正所有无效案例

        Args:
            invalid_cases: 无效案例列表
            tool: 工具对象

        Returns:
            修正后的案例列表
        """
        print("\n" + "=" * 80)
        print("🔧 案例修正器")
        print("=" * 80)

        # 只修正judge=2的案例（需要辅助的案例）
        cases_to_revise = [c for c in invalid_cases if c.check_result.get('judge') == 2]
        print(f"待修正案例数: {len(cases_to_revise)}")

        if not cases_to_revise:
            print("ℹ️  没有需要修正的案例（judge=2）")
            return []

        revised_cases = []
        tool_dict = tool.to_dict()

        for i, case in enumerate(cases_to_revise, 1):
            print(f"\n[{i}/{len(cases_to_revise)}] 修正案例 {case.case_id}...", end=" ")

            try:
                # 使用Langchain工具链修正单个案例
                check_result = case.check_result or {}
                constructive_criticism = check_result.get('evidence', '未提供批评意见')

                revised_text = revise_case_with_llm(
                    case_text=case.generated_case,
                    scenario_name=case.scenario_name,
                    tool=tool_dict,
                    constructive_criticism=constructive_criticism,
                    llm=self.llm
                )

                # 记录修正信息
                revision_info = {
                    "original_case": case.generated_case,
                    "revised_case": revised_text,
                    "invalid_reason": check_result.get('evidence', ''),
                    "revision_timestamp": datetime.now().isoformat()
                }

                # 更新案例
                case.generated_case = revised_text
                case.status = CaseStatus.REVISED
                case.revision_result = revision_info

                revised_cases.append(case)
                print("✅ 修正完成")

            except Exception as e:
                print(f"❌ 修正失败: {e}")
                case.status = CaseStatus.FAILED
                case.revision_result = {"error": str(e)}

        print(f"\n✅ 修正完成: {len(revised_cases)}/{len(cases_to_revise)}")

        # 保存结果
        if self.output_dir:
            self._save_revised_cases(revised_cases)

        return revised_cases

    def _save_revised_cases(self, revised_cases: List[Case]):
        """
        保存修正后的案例（已禁用，改由workflow_orchestrator统一保存）

        Args:
            revised_cases: 修正后的案例列表
        """
        # 不再保存单独的revised_cases.json
        # 这些数据会在workflow_orchestrator中以兼容旧格式的方式保存为revision_result_*.json
        pass
