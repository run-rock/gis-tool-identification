#!/usr/bin/env python3
"""
工作流编排器 - 协调整个案例生成流程
"""

import os
import json
from typing import Dict, List
from datetime import datetime, timedelta

from ..models import Tool, Toolbox, WorkflowResult, ProcessStatus
from .scenario_generator import ScenarioGenerator
from .case_generator import CaseGenerator
from .case_validator import CaseValidator
from .case_reviser import CaseReviser


class Stage1Orchestrator:
    """
    工作流编排器

    职责：协调场景生成、案例生成、案例验证、案例修正四个步骤
    """

    def __init__(
        self,
        scenario_llm,
        case_llm,
        check_llm,
        revise_llm,
        output_dir: str,
        num_scenarios: int = 5,
        cases_per_scenario: int = 5,
        with_parameters: bool = False,
        enable_revision: bool = True
    ):
        """
        初始化工作流编排器

        Args:
            scenario_llm: 场景生成LLM（建议temperature=0.6）
            case_llm: 案例生成LLM（建议temperature=0.6）
            check_llm: 案例验证LLM（建议temperature=0.3）
            revise_llm: 案例修正LLM（建议temperature=0.7）
            output_dir: 输出目录
            num_scenarios: 生成场景数量
            cases_per_scenario: 每个场景生成的案例数
            with_parameters: 是否在案例生成时包含工具参数
            enable_revision: 是否启用案例修正
        """
        self.output_dir = output_dir
        self.num_scenarios = num_scenarios
        self.cases_per_scenario = cases_per_scenario
        self.with_parameters = with_parameters
        self.enable_revision = enable_revision

        # 创建各个工作流组件
        self.scenario_generator = ScenarioGenerator(
            llm=scenario_llm,
            output_dir=None,  # 子组件不单独保存，由编排器统一管理
            num_scenarios=num_scenarios
        )

        self.case_generator = CaseGenerator(
            llm=case_llm,
            output_dir=None,
            cases_per_scenario=cases_per_scenario,
            with_parameters=with_parameters
        )

        self.case_validator = CaseValidator(
            llm=check_llm,
            output_dir=None
        )

        self.case_reviser = CaseReviser(
            llm=revise_llm,
            output_dir=None
        )

        # 工作流追踪信息
        self.workflow_start_time = None
        self.files_generated = []
        self.steps_completed = []

        print("🚀 工作流编排器已初始化")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 配置: {num_scenarios}场景 × {cases_per_scenario}案例 = {num_scenarios * cases_per_scenario}总案例")

    def run(
        self,
        tool: Tool,
        toolbox: Toolbox
    ) -> WorkflowResult:
        """
        运行完整工作流

        Args:
            tool: 工具对象
            toolbox: 工具箱对象

        Returns:
            WorkflowResult对象
        """
        print("\n" + "=" * 80)
        print(f"🎯 开始工作流: {tool.name}")
        print("=" * 80)

        # 记录开始时间
        self.workflow_start_time = datetime.now()
        self.files_generated = []
        self.steps_completed = []

        # 创建工具专用输出目录
        tool_output_dir = self._setup_output_dir(tool)

        # 设置所有组件的输出目录
        self.scenario_generator.output_dir = tool_output_dir
        self.case_generator.output_dir = tool_output_dir
        self.case_validator.output_dir = tool_output_dir
        self.case_reviser.output_dir = tool_output_dir

        try:
            # 步骤1: 生成场景
            scenarios = self.scenario_generator.execute(
                tool=tool,
                toolbox=toolbox
            )
            self.steps_completed.append("generate_scenarios")
            scenario_file = os.path.join(tool_output_dir, f"scenarios_{tool.algorithm_id.replace(':', '_')}.json")
            self.files_generated.append(scenario_file)

            # 步骤2: 生成案例
            cases = self.case_generator.execute(
                tool=tool,
                toolbox=toolbox,
                scenarios=scenarios
            )
            self.steps_completed.append("generate_cases")
            cases_file = os.path.join(tool_output_dir, f"cases_{tool.algorithm_id.replace(':', '_')}.json")
            self.files_generated.append(cases_file)

            # 步骤3: 验证案例
            validation_results = self.case_validator.execute(
                cases=cases,
                tool=tool
            )

            valid_cases = validation_results["valid"]
            invalid_cases = validation_results["invalid"]

            # 保存初始检查结果 (check_result_cases_*.json)
            check_file = self._save_check_result(
                tool=tool,
                toolbox=toolbox,
                cases=cases,
                output_dir=tool_output_dir,
                file_prefix=f"cases_{tool.algorithm_id.replace(':', '_')}"
            )
            self.steps_completed.append("check_cases")
            self.files_generated.append(check_file)

            # 步骤4: 修正无效案例（可选）
            revision_file = None
            final_check_file = None
            revised_valid_cases = []

            if self.enable_revision and invalid_cases:
                # 修正无效案例
                revised_cases = self.case_reviser.execute(
                    invalid_cases=invalid_cases,
                    tool=tool
                )

                # 保存修正结果 (revision_result_check_result_cases_*.json)
                if revised_cases:
                    revision_file = self._save_revision_result(
                        tool=tool,
                        toolbox=toolbox,
                        check_file=check_file,
                        revised_cases=revised_cases,
                        output_dir=tool_output_dir
                    )
                    self.steps_completed.append("revise_cases")
                    self.files_generated.append(revision_file)

                    # 再次验证修正后的案例
                    print("\n🔍 再次验证修正后的案例...")
                    re_validation_results = self.case_validator.execute(
                        cases=revised_cases,
                        tool=tool
                    )

                    revised_valid_cases = re_validation_results["valid"]
                    revised_invalid_cases = re_validation_results["invalid"]

                    # 保存最终检查结果 (final_check_revision_result_check_result_cases_*.json)
                    final_check_file = self._save_final_check_result(
                        tool=tool,
                        toolbox=toolbox,
                        revised_cases=revised_cases,
                        re_validation_results=re_validation_results,
                        output_dir=tool_output_dir,
                        revision_file=revision_file
                    )
                    self.steps_completed.append("final_check")
                    self.files_generated.append(final_check_file)

                    # 更新有效案例列表
                    valid_cases.extend(revised_valid_cases)

            # 统计结果
            all_cases = valid_cases + invalid_cases + revised_valid_cases
            statistics = self._calculate_statistics(
                scenarios=scenarios,
                all_cases=all_cases,
                valid_cases=valid_cases,
                invalid_cases=invalid_cases,
                revised_cases=revised_valid_cases
            )

            # 创建WorkflowResult
            result = WorkflowResult(
                tool=tool,
                toolbox=toolbox,
                scenarios=scenarios,
                cases=all_cases,
                statistics=statistics,
                status=ProcessStatus.SUCCESS
            )

            # 保存工作流报告 (workflow_report_*.json)
            self._save_workflow_report(
                tool=tool,
                toolbox=toolbox,
                output_dir=tool_output_dir,
                scenario_file=scenario_file,
                cases_file=cases_file,
                check_file=check_file,
                revision_file=revision_file,
                final_check_file=final_check_file,
                status="completed"
            )

            print("\n" + "=" * 80)
            print("✅ 工作流完成")
            print("=" * 80)
            self._print_summary(statistics)

            return result

        except Exception as e:
            print(f"\n❌ 工作流失败: {e}")
            result = WorkflowResult(
                tool=tool,
                toolbox=toolbox,
                scenarios=[],
                cases=[],
                statistics={},
                status=ProcessStatus.FAILED,
                error_message=str(e)
            )
            return result

    def _setup_output_dir(self, tool: Tool) -> str:
        """
        创建工具专用输出目录

        Args:
            tool: 工具对象

        Returns:
            工具输出目录路径
        """
        tool_id = tool.algorithm_id.replace(':', '_')
        tool_output_dir = os.path.join(self.output_dir, tool_id)
        os.makedirs(tool_output_dir, exist_ok=True)
        return tool_output_dir

    def _calculate_statistics(
        self,
        scenarios,
        all_cases,
        valid_cases,
        invalid_cases,
        revised_cases
    ) -> Dict:
        """
        计算统计信息

        Args:
            scenarios: 场景列表
            all_cases: 所有案例
            valid_cases: 有效案例
            invalid_cases: 无效案例
            revised_cases: 修正后的案例

        Returns:
            统计字典
        """
        total_cases = len(all_cases)
        valid_count = len(valid_cases)
        invalid_count = len(invalid_cases)
        revised_count = len(revised_cases)

        return {
            "total_scenarios": len(scenarios),
            "total_cases": total_cases,
            "valid_cases": valid_count,
            "invalid_cases": invalid_count,
            "revised_cases": revised_count,
            "valid_rate": valid_count / total_cases * 100 if total_cases > 0 else 0,
            "invalid_rate": invalid_count / total_cases * 100 if total_cases > 0 else 0,
            "revision_enabled": self.enable_revision
        }

    def _save_check_result(
        self,
        tool: Tool,
        toolbox: Toolbox,
        cases: List,
        output_dir: str,
        file_prefix: str
    ) -> str:
        """
        保存检查结果（兼容旧格式）

        Args:
            tool: 工具对象
            toolbox: 工具箱对象
            cases: 案例列表（Case对象列表）
            output_dir: 输出目录
            file_prefix: 文件前缀

        Returns:
            检查结果文件路径
        """
        # 构建checked_cases列表
        checked_cases = []
        valid_count = 0
        invalid_count = 0
        failed_count = 0

        for i, case in enumerate(cases, 1):
            # Case 是对象，需要访问属性而不是字典键
            check_result = case.check_result or {}
            is_valid = check_result.get('is_valid', False)

            if is_valid:
                valid_count += 1
            elif check_result.get('judge', 1) == -1:
                failed_count += 1
            else:
                invalid_count += 1

            checked_case = {
                "case_id": i,
                "original_case": {
                    "generated_case": case.generated_case,
                    "scenario": {"name": case.scenario_name}
                },
                "check_result": {
                    "judge": check_result.get('judge', -1),
                    "evidence": check_result.get('evidence', ''),
                    "is_valid": is_valid
                },
                "check_timestamp": datetime.now().isoformat()
            }
            checked_cases.append(checked_case)

        # 构建检查报告（兼容旧格式）
        report = {
            "check_summary": {
                "total_cases": len(cases),
                "valid_cases": valid_count,
                "invalid_cases": invalid_count,
                "failed_checks": failed_count,
                "validity_rate": valid_count / len(cases) * 100 if cases else 0
            },
            "tool_info": tool.to_dict(),
            "toolbox_info": toolbox.to_dict(),
            "check_timestamp": datetime.now().isoformat(),
            "checked_cases": checked_cases
        }

        # 保存文件
        filename = f"check_result_{file_prefix}.json"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"💾 检查结果已保存: {file_path}")
        return file_path

    def _save_revision_result(
        self,
        tool: Tool,
        toolbox: Toolbox,
        check_file: str,
        revised_cases: List,
        output_dir: str
    ) -> str:
        """
        保存修正结果（兼容旧格式）

        Args:
            tool: 工具对象
            toolbox: 工具箱对象
            check_file: 初始检查文件路径
            revised_cases: 修正后的案例列表（Case对象列表）
            output_dir: 输出目录

        Returns:
            修正结果文件路径
        """
        # 构建revised_cases列表
        revised_cases_data = []
        success_count = 0
        fail_count = 0

        for i, case in enumerate(revised_cases, 1):
            # Case 是对象，访问属性
            revision_result = case.revision_result or {}
            original_check = case.check_result or {}

            if 'error' not in revision_result and revision_result.get('revised_case'):
                success_count += 1
            else:
                fail_count += 1

            # 获取原始案例文本（修正前的）
            original_case_text = revision_result.get('original_case', case.generated_case)

            revised_case_data = {
                "original_case_id": case.case_id,
                "original_case_info": {
                    "generated_case": original_case_text,
                    "scenario": {"name": case.scenario_name}
                },
                "original_check_result": {
                    "judge": original_check.get('judge', 0),
                    "evidence": original_check.get('evidence', ''),
                    "is_valid": False
                },
                "revision_result": {
                    "revised_case": revision_result.get('revised_case', ''),
                    "constructive_criticism_used": original_check.get('evidence', '')
                },
                "revision_timestamp": revision_result.get('revision_timestamp', datetime.now().isoformat())
            }

            if 'error' in revision_result:
                revised_case_data["revision_result"]["error"] = revision_result["error"]

            revised_cases_data.append(revised_case_data)

        # 构建修正报告（兼容旧格式）
        report = {
            "revision_summary": {
                "input_file": check_file,
                "total_cases_to_revise": len(revised_cases),
                "successfully_revised": success_count,
                "failed_revisions": fail_count,
                "revision_success_rate": success_count / len(revised_cases) * 100 if revised_cases else 0
            },
            "tool_info": tool.to_dict(),
            "toolbox_info": toolbox.to_dict(),
            "revision_timestamp": datetime.now().isoformat(),
            "revised_cases": revised_cases_data
        }

        # 保存文件
        filename = f"revision_result_{os.path.basename(check_file)}"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"💾 修正结果已保存: {file_path}")
        return file_path

    def _save_final_check_result(
        self,
        tool: Tool,
        toolbox: Toolbox,
        revised_cases: List,
        re_validation_results: Dict,
        output_dir: str,
        revision_file: str
    ) -> str:
        """
        保存最终检查结果（兼容旧格式）

        Args:
            tool: 工具对象
            toolbox: 工具箱对象
            revised_cases: 修正后的案例列表（Case对象列表）
            re_validation_results: 重新验证结果
            output_dir: 输出目录
            revision_file: 修正结果文件路径

        Returns:
            最终检查结果文件路径
        """
        # 构建checked_cases列表（针对修正后的案例）
        checked_cases = []
        valid_count = len(re_validation_results.get('valid', []))
        invalid_count = len(re_validation_results.get('invalid', []))
        failed_count = 0

        # 合并有效和无效案例
        all_re_validated = re_validation_results.get('valid', []) + re_validation_results.get('invalid', [])

        for i, case in enumerate(all_re_validated, 1):
            # Case 是对象，访问属性
            check_result = case.check_result or {}
            is_valid = check_result.get('is_valid', False)

            if check_result.get('judge', 1) == -1:
                failed_count += 1

            checked_case = {
                "case_id": i,
                "original_case": {
                    "generated_case": case.generated_case,
                    "scenario": {"name": case.scenario_name}
                },
                "check_result": {
                    "judge": check_result.get('judge', -1),
                    "evidence": check_result.get('evidence', ''),
                    "is_valid": is_valid
                },
                "check_timestamp": datetime.now().isoformat()
            }
            checked_cases.append(checked_case)

        # 构建最终检查报告（兼容旧格式）
        report = {
            "check_summary": {
                "total_cases": len(all_re_validated),
                "valid_cases": valid_count,
                "invalid_cases": invalid_count,
                "failed_checks": failed_count,
                "validity_rate": valid_count / len(all_re_validated) * 100 if all_re_validated else 0
            },
            "tool_info": tool.to_dict(),
            "toolbox_info": toolbox.to_dict(),
            "check_timestamp": datetime.now().isoformat(),
            "checked_cases": checked_cases
        }

        # 保存文件
        filename = f"final_check_{os.path.basename(revision_file)}"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"💾 最终检查结果已保存: {file_path}")
        return file_path

    def _save_workflow_report(
        self,
        tool: Tool,
        toolbox: Toolbox,
        output_dir: str,
        scenario_file: str,
        cases_file: str,
        check_file: str,
        revision_file: str,
        final_check_file: str,
        status: str
    ):
        """
        保存工作流报告（兼容旧格式）

        Args:
            tool: 工具对象
            toolbox: 工具箱对象
            output_dir: 输出目录
            scenario_file: 场景文件路径
            cases_file: 案例文件路径
            check_file: 检查文件路径
            revision_file: 修正文件路径（可为None）
            final_check_file: 最终检查文件路径（可为None）
            status: 工作流状态
        """
        workflow_end_time = datetime.now()
        total_duration = workflow_end_time - self.workflow_start_time

        report = {
            "tool": tool.to_dict(),
            "toolbox": toolbox.to_dict(),
            "workflow_start_time": self.workflow_start_time.isoformat(),
            "steps_completed": self.steps_completed,
            "files_generated": self.files_generated,
            "scenario_file": scenario_file,
            "cases_file": cases_file,
            "check_file": check_file,
            "revision_file": revision_file,
            "final_check_file": final_check_file,
            "workflow_end_time": workflow_end_time.isoformat(),
            "total_duration": str(total_duration),
            "status": status
        }

        # 保存文件
        filename = f"workflow_report_{tool.algorithm_id.replace(':', '_')}.json"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"💾 工作流报告已保存: {file_path}")

    def _print_summary(self, statistics: Dict):
        """
        打印统计摘要

        Args:
            statistics: 统计字典
        """
        print(f"\n📊 统计摘要:")
        print(f"  场景数: {statistics['total_scenarios']}")
        print(f"  总案例数: {statistics['total_cases']}")
        print(f"  ✅ 有效: {statistics['valid_cases']} ({statistics['valid_rate']:.1f}%)")
        print(f"  ❌ 无效: {statistics['invalid_cases']} ({statistics['invalid_rate']:.1f}%)")
        if statistics['revision_enabled']:
            print(f"  🔧 已修正: {statistics['revised_cases']}")
