#!/usr/bin/env python3
"""
案例验证器 - 验证生成的案例是否符合要求
"""

from typing import List, Dict
from datetime import datetime

from ..base import BaseWorkflow
from ..models import Case, CaseStatus, Tool
from ..services.llm_utils import validate_case_with_llm


class CaseValidator(BaseWorkflow):
    """
    案例验证器

    职责：验证案例是否符合质量要求
    """

    def __init__(self, llm, output_dir: str = None):
        """
        初始化案例验证器

        Args:
            llm: LLM模型实例（建议temperature=0.3，更一致的判断）
            output_dir: 输出目录
        """
        super().__init__(llm, output_dir)

    def execute(self, cases: List[Case], tool: Tool) -> Dict[str, List[Case]]:
        """
        验证所有案例

        Args:
            cases: 案例列表
            tool: 工具对象

        Returns:
            验证结果字典 {"valid": [...], "invalid": [...]}
        """
        print("\n" + "=" * 80)
        print("🔍 案例验证器")
        print("=" * 80)
        print(f"待验证案例数: {len(cases)}")

        valid_cases = []
        invalid_cases = []

        tool_dict = tool.to_dict()

        for i, case in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}] 验证案例 {case.case_id}...", end=" ")

            try:
                # 使用Langchain工具链验证单个案例
                check_result = validate_case_with_llm(
                    case_text=case.generated_case,
                    tool=tool_dict,
                    llm=self.llm
                )

                # 更新案例状态
                case.check_result = check_result

                if check_result['is_valid']:
                    case.status = CaseStatus.VALID
                    valid_cases.append(case)
                    print("✅ 有效")
                else:
                    case.status = CaseStatus.INVALID
                    invalid_cases.append(case)
                    # judge: 0=无法解决, 2=需要辅助
                    judge = check_result.get('judge', 0)
                    if judge == 2:
                        print(f"❌ 无效（需要辅助）")
                    else:
                        print(f"❌ 无效（无法解决）")

            except Exception as e:
                print(f"⚠️  验证失败: {e}")
                case.status = CaseStatus.FAILED
                case.check_result = {"error": str(e), "is_valid": False}
                invalid_cases.append(case)

        # 打印统计
        print(f"\n{'='*80}")
        print(f"验证完成:")
        print(f"  ✅ 有效: {len(valid_cases)} ({len(valid_cases)/len(cases)*100:.1f}%)")
        print(f"  ❌ 无效: {len(invalid_cases)} ({len(invalid_cases)/len(cases)*100:.1f}%)")
        print(f"{'='*80}")

        # 保存结果
        if self.output_dir:
            self._save_validation_results(valid_cases, invalid_cases)

        return {
            "valid": valid_cases,
            "invalid": invalid_cases
        }

    def _save_validation_results(self, valid_cases: List[Case], invalid_cases: List[Case]):
        """
        保存验证结果（已禁用，改由workflow_orchestrator统一保存）

        Args:
            valid_cases: 有效案例列表
            invalid_cases: 无效案例列表
        """
        # 不再保存单独的valid_cases.json和invalid_cases.json
        # 这些数据会在workflow_orchestrator中以兼容旧格式的方式保存为check_result_cases_*.json
        pass
