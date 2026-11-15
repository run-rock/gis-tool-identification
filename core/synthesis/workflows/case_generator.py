#!/usr/bin/env python3
"""
案例生成器 - 根据场景生成具体案例
"""

from typing import Dict, List
from datetime import datetime

from ..base import BaseWorkflow
from ..models import Tool, Toolbox, Scenario, Case, CaseStatus
from ..services.llm_utils import generate_case_with_llm


class CaseGenerator(BaseWorkflow):
    """
    案例生成器

    职责：根据场景生成具体的应用案例
    """

    def __init__(self, llm, output_dir: str = None, cases_per_scenario: int = 5, with_parameters: bool = False):
        """
        初始化案例生成器

        Args:
            llm: LLM模型实例（建议temperature=0.6）
            output_dir: 输出目录
            cases_per_scenario: 每个场景生成的案例数
            with_parameters: 是否在prompt中包含工具参数信息
        """
        super().__init__(llm, output_dir)
        self.cases_per_scenario = cases_per_scenario
        self.with_parameters = with_parameters

    def execute(
        self,
        tool: Tool,
        toolbox: Toolbox,
        scenarios: List[Scenario]
    ) -> List[Case]:
        """
        为所有场景生成案例

        Args:
            tool: 工具对象
            toolbox: 工具箱对象
            scenarios: 场景列表

        Returns:
            案例列表
        """
        print("\n" + "=" * 80)
        print("📋 案例生成器")
        print("=" * 80)
        print(f"工具: {tool.name}")
        print(f"场景数: {len(scenarios)}")
        print(f"每场景案例数: {self.cases_per_scenario}")
        print(f"总目标案例数: {len(scenarios) * self.cases_per_scenario}")

        all_cases = []
        case_id = 1

        # 准备工具字典（根据with_parameters决定是否包含参数）
        tool_dict = tool.to_dict()
        if not self.with_parameters:
            tool_dict = {
                'name': tool.name,
                'description': tool.description,
                'algorithm_id': tool.algorithm_id
            }

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n--- 场景 {i}/{len(scenarios)}: {scenario.name} ---")

            for case_num in range(1, self.cases_per_scenario + 1):
                print(f"  生成案例 {case_num}/{self.cases_per_scenario}...", end=" ")

                try:
                    # 使用Langchain工具链生成单个案例
                    generated_text = generate_case_with_llm(
                        tool=tool_dict,
                        scenario=scenario.to_dict(),
                        llm=self.llm
                    )

                    # 创建Case对象
                    case = Case(
                        case_id=case_id,
                        scenario_name=scenario.name,
                        generated_case=generated_text,
                        status=CaseStatus.PENDING,
                        metadata={
                            "scenario_id": scenario.scenario_id,
                            "generation_timestamp": datetime.now().isoformat()
                        }
                    )

                    all_cases.append(case)
                    case_id += 1
                    print("✅")

                except Exception as e:
                    print(f"❌ 失败: {e}")

        print(f"\n✅ 共生成 {len(all_cases)} 个案例")

        # 保存结果
        if self.output_dir:
            self._save_cases(tool, all_cases)

        return all_cases


    def _save_cases(self, tool: Tool, cases: List[Case]):
        """
        保存案例到文件

        Args:
            tool: 工具对象
            cases: 案例列表
        """
        data = {
            "tool": tool.to_dict(),
            "total_cases": len(cases),
            "generation_timestamp": datetime.now().isoformat(),
            "cases": [c.to_dict() for c in cases]
        }

        filename = f"cases_{tool.algorithm_id.replace(':', '_')}.json"
        file_path = self.save_json(data, filename)
        print(f"💾 案例已保存: {file_path}")
