#!/usr/bin/env python3
"""
场景生成器 - 根据工具和工具箱生成应用场景
"""

from typing import Dict, List
from datetime import datetime

from ..base import BaseWorkflow
from ..models import Tool, Toolbox, Scenario
from ..services.llm_utils import generate_scenarios_with_llm


class ScenarioGenerator(BaseWorkflow):
    """
    场景生成器

    职责：根据工具和工具箱信息生成多个应用场景
    """

    def __init__(self, llm, output_dir: str = None, num_scenarios: int = 5):
        """
        初始化场景生成器

        Args:
            llm: LLM模型实例（建议temperature=0.6）
            output_dir: 输出目录
            num_scenarios: 生成场景数量
        """
        super().__init__(llm, output_dir)
        self.num_scenarios = num_scenarios

    def execute(self, tool: Tool, toolbox: Toolbox) -> List[Scenario]:
        """
        生成场景

        Args:
            tool: 工具对象
            toolbox: 工具箱对象

        Returns:
            场景列表
        """
        print("\n" + "=" * 80)
        print("📝 场景生成器")
        print("=" * 80)
        print(f"工具: {tool.name}")
        print(f"工具箱: {toolbox.name}")
        print(f"目标场景数: {self.num_scenarios}")

        try:
            # 调用LLM生成场景（使用Langchain工具链）
            print("\n⏳ 调用LLM生成场景...")
            result = generate_scenarios_with_llm(
                tool=tool.to_dict(),
                llm=self.llm
            )

            # 解析响应，转换为Scenario对象
            scenarios = self._parse_response(result['scenarios'])

            print(f"\n✅ 成功生成 {len(scenarios)} 个场景:")
            for i, scenario in enumerate(scenarios, 1):
                print(f"  {i}. {scenario.name}")

            # 保存结果
            if self.output_dir:
                self._save_scenarios(tool, toolbox, scenarios)

            return scenarios

        except Exception as e:
            print(f"\n❌ 场景生成失败: {e}")
            raise

    def _parse_response(self, scenario_data: List[Dict]) -> List[Scenario]:
        """
        解析场景数据，转换为Scenario对象列表

        Args:
            scenario_data: 从LLM返回的场景数据列表

        Returns:
            场景对象列表
        """
        scenarios = []
        for i, item in enumerate(scenario_data, 1):
            scenario = Scenario(
                name=item.get('name', f"场景{i}"),
                description=item.get('description', ''),
                scenario_id=i
            )
            scenarios.append(scenario)
        return scenarios

    def _save_scenarios(self, tool: Tool, toolbox: Toolbox, scenarios: List[Scenario]):
        """
        保存场景到文件

        Args:
            tool: 工具对象
            toolbox: 工具箱对象
            scenarios: 场景列表
        """
        data = {
            "tool": tool.to_dict(),
            "toolbox": toolbox.to_dict(),
            "total_scenarios": len(scenarios),
            "generation_timestamp": datetime.now().isoformat(),
            "scenarios": [s.to_dict() for s in scenarios]
        }

        filename = f"scenarios_{tool.algorithm_id.replace(':', '_')}.json"
        file_path = self.save_json(data, filename)
        print(f"💾 场景已保存: {file_path}")
