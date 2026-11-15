#!/usr/bin/env python3
"""
LLM工具函数 - 封装Langchain的PromptTemplate和JsonOutputParser

提供统一的接口用于场景生成、案例生成、案例验证和案例修正
"""

from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from ..prompts import (
    ScenarioList,
    GeneratedCase,
    CaseCheck,
    RevisedCase,
    SingleToolCheck1,
    SingleToolCheck2,
    SCENARIO_PROMPT,
    GENERATE_CASE_FROM_SCENARIO_PROMPT,
    CHECK_CASE_PROMPT,
    REVISE_CASE_PROMPT,
    POSITIVE_SAMPLE1_PROMPT,
    POSITIVE_SAMPLE2_PROMPT
)


def _should_use_no_think(llm) -> bool:
    """
    检测LLM是否应该使用/no_think机制

    针对Qwen3模型（如qwen3-32b-gguf），在prompt末尾添加/no_think可以禁用思考过程

    Args:
        llm: LLM模型实例

    Returns:
        是否应该使用no_think
    """
    # 尝试获取model_name属性
    model_name = getattr(llm, 'model_name', '').lower()

    # 检查是否为qwen3模型
    if 'qwen3' in model_name:
        return True

    return False


def _apply_no_think_if_needed(template: str, llm) -> str:
    """
    如果需要，在prompt模板末尾添加/no_think

    Args:
        template: 原始prompt模板
        llm: LLM模型实例

    Returns:
        处理后的prompt模板
    """
    if _should_use_no_think(llm):
        # 在模板末尾添加/no_think（保留换行）
        if not template.rstrip().endswith('/no_think'):
            return template.rstrip() + ' /no_think\n'

    return template


def generate_scenarios_with_llm(tool: Dict, llm, max_retries: int = 3) -> Dict:
    """
    使用LLM生成场景

    Args:
        tool: 工具信息字典
        llm: LLM模型实例（如果是Qwen3，会自动添加/no_think）
        max_retries: 最大重试次数

    Returns:
        场景生成结果字典
    """
    # 初始化输出解析器
    parser = JsonOutputParser(pydantic_object=ScenarioList)

    # 应用no_think机制（如果需要）
    template = _apply_no_think_if_needed(SCENARIO_PROMPT, llm)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=template,
        input_variables=["gis_tool"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 构建Chain
    chain = prompt | llm | parser

    # 带重试机制的生成
    last_error = None
    for attempt in range(max_retries):
        try:
            result = chain.invoke({"gis_tool": tool})

            # 检查result是否为None或缺少必需字段
            if result is None:
                raise ValueError("LLM返回了空结果")

            if "scenarios" not in result:
                raise ValueError(f"LLM返回的JSON缺少'scenarios'字段，返回内容: {result}")

            # 成功，返回结果
            return result

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"⚠️  场景生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"🔄 正在重试...")
            else:
                print(f"❌ 场景生成失败，已达到最大重试次数")

    # 所有重试都失败
    raise ValueError(f"场景生成失败（重试{max_retries}次后）: {last_error}")


def generate_case_with_llm(tool: Dict, scenario: Dict, llm, max_retries: int = 3) -> str:
    """
    使用LLM生成案例

    Args:
        tool: 工具信息字典
        scenario: 场景信息字典
        llm: LLM模型实例（如果是Qwen3，会自动添加/no_think）
        max_retries: 最大重试次数

    Returns:
        生成的案例文本
    """
    # 初始化输出解析器
    parser = JsonOutputParser(pydantic_object=GeneratedCase)

    # 应用no_think机制（如果需要）
    template = _apply_no_think_if_needed(GENERATE_CASE_FROM_SCENARIO_PROMPT, llm)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=template,
        input_variables=["gis_tool", "scenario"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 构建Chain
    chain = prompt | llm | parser

    # 剔除scenario中的tool_use字段（仅保留name，避免泄露工具信息）
    scenario_without_tool_use = {
        "name": scenario.get('name', '')
    }

    # 带重试机制的生成
    last_error = None
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "gis_tool": tool,
                "scenario": scenario_without_tool_use
            })

            # 检查result是否为None或缺少必需字段
            if result is None:
                raise ValueError("LLM返回了空结果")

            if "case" not in result:
                raise ValueError(f"LLM返回的JSON缺少'case'字段，返回内容: {result}")

            # 成功，返回案例文本
            return result['case']

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"⚠️  案例生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"🔄 正在重试...")
            else:
                print(f"❌ 案例生成失败，已达到最大重试次数")

    # 所有重试都失败
    raise ValueError(f"案例生成失败（重试{max_retries}次后）: {last_error}")


def validate_case_with_llm(case_text: str, tool: Dict, llm, max_retries: int = 3) -> Dict:
    """
    使用LLM验证案例

    Args:
        case_text: 案例文本
        tool: 工具信息字典
        llm: LLM模型实例（如果是Qwen3，会自动添加/no_think）
        max_retries: 最大重试次数

    Returns:
        验证结果字典 {"judge": int, "evidence": str}
    """
    # 初始化输出解析器
    parser = JsonOutputParser(pydantic_object=CaseCheck)

    # 应用no_think机制（如果需要）
    template = _apply_no_think_if_needed(CHECK_CASE_PROMPT, llm)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=template,
        input_variables=["case", "gis_tool"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 构建Chain
    chain = prompt | llm | parser

    # 带重试机制的验证
    last_error = None
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "case": case_text,
                "gis_tool": tool
            })

            # 检查result是否为None或缺少必需字段
            if result is None:
                raise ValueError("LLM返回了空结果")

            if "judge" not in result or "evidence" not in result:
                raise ValueError(f"LLM返回的JSON缺少必需字段，返回内容: {result}")

            # 成功，返回验证结果
            return {
                'judge': result['judge'],
                'evidence': result['evidence'],
                'is_valid': result['judge'] == 1  # judge=1表示有效
            }

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"⚠️  案例验证失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"🔄 正在重试...")
            else:
                print(f"❌ 案例验证失败，已达到最大重试次数")

    # 所有重试都失败
    raise ValueError(f"案例验证失败（重试{max_retries}次后）: {last_error}")


def revise_case_with_llm(
    case_text: str,
    scenario_name: str,
    tool: Dict,
    constructive_criticism: str,
    llm,
    max_retries: int = 3
) -> str:
    """
    使用LLM修正案例

    Args:
        case_text: 原始案例文本
        scenario_name: 场景名称
        tool: 工具信息字典
        constructive_criticism: 建设性批评意见
        llm: LLM模型实例（如果是Qwen3，会自动添加/no_think）
        max_retries: 最大重试次数

    Returns:
        修正后的案例文本
    """
    # 初始化输出解析器
    parser = JsonOutputParser(pydantic_object=RevisedCase)

    # 应用no_think机制（如果需要）
    template = _apply_no_think_if_needed(REVISE_CASE_PROMPT, llm)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=template,
        input_variables=["case", "scenario", "gis_tool", "constructive_criticism"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 构建Chain
    chain = prompt | llm | parser

    # 带重试机制的修正
    last_error = None
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "case": case_text,
                "scenario": scenario_name,
                "gis_tool": tool,
                "constructive_criticism": constructive_criticism
            })

            # 检查result是否为None或缺少必需字段
            if result is None:
                raise ValueError("LLM返回了空结果")

            if "revised_case" not in result:
                raise ValueError(f"LLM返回的JSON缺少'revised_case'字段，返回内容: {result}")

            # 成功，返回修正后的案例文本
            return result['revised_case']

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"⚠️  案例修正失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"🔄 正在重试...")
            else:
                print(f"❌ 案例修正失败，已达到最大重试次数")

    # 所有重试都失败
    raise ValueError(f"案例修正失败（重试{max_retries}次后）: {last_error}")


# ============================================================================
# 第二阶段：相似工具评估
# ============================================================================

def evaluate_similar_tool_stage1(
    case_text: str,
    tool: Dict,
    llm,
    max_retries: int = 3
) -> Dict:
    """
    阶段1：评估工具的数据和功能匹配性

    Args:
        case_text: 案例文本
        tool: 工具信息字典（包含name, algorithm_id, description, parameters）
        llm: LLM模型实例（如果是Qwen3，会自动添加/no_think）
        max_retries: 最大重试次数

    Returns:
        评估结果字典 {"judge": int, "evidence": str}
    """
    # 初始化输出解析器
    parser = JsonOutputParser(pydantic_object=SingleToolCheck1)

    # 应用no_think机制（如果需要）
    template = _apply_no_think_if_needed(POSITIVE_SAMPLE1_PROMPT, llm)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=template,
        input_variables=["case", "gis_tool"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 构建Chain
    chain = prompt | llm | parser

    # 带重试机制的评估
    last_error = None
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "case": case_text,
                "gis_tool": tool
            })

            # 检查result是否为None或缺少必需字段
            if result is None:
                raise ValueError("LLM返回了空结果")

            if "judge" not in result or "evidence" not in result:
                raise ValueError(f"LLM返回的JSON缺少必需字段，返回内容: {result}")

            # 成功，返回评估结果
            return {
                'judge': result['judge'],
                'evidence': result['evidence']
            }

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"      ⚠️  阶段1评估失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"      🔄 正在重试...")

    # 所有重试都失败
    return {
        'judge': -1,
        'evidence': f"阶段1评估失败（重试{max_retries}次后）: {last_error}"
    }


def evaluate_similar_tool_stage2(
    case_text: str,
    tool: Dict,
    llm,
    max_retries: int = 3
) -> Dict:
    """
    阶段2：评估工具的输出是否满足需求

    Args:
        case_text: 案例文本
        tool: 工具信息字典（包含name, algorithm_id, description, parameters）
        llm: LLM模型实例（如果是Qwen3，会自动添加/no_think）
        max_retries: 最大重试次数

    Returns:
        评估结果字典 {"judge": int, "evidence": str}
    """
    # 初始化输出解析器
    parser = JsonOutputParser(pydantic_object=SingleToolCheck2)

    # 应用no_think机制（如果需要）
    template = _apply_no_think_if_needed(POSITIVE_SAMPLE2_PROMPT, llm)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=template,
        input_variables=["case", "gis_tool"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 构建Chain
    chain = prompt | llm | parser

    # 带重试机制的评估
    last_error = None
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "case": case_text,
                "gis_tool": tool
            })

            # 检查result是否为None或缺少必需字段
            if result is None:
                raise ValueError("LLM返回了空结果")

            if "judge" not in result or "evidence" not in result:
                raise ValueError(f"LLM返回的JSON缺少必需字段，返回内容: {result}")

            # 成功，返回评估结果
            return {
                'judge': result['judge'],
                'evidence': result['evidence']
            }

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"      ⚠️  阶段2评估失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"      🔄 正在重试...")

    # 所有重试都失败
    return {
        'judge': -1,
        'evidence': f"阶段2评估失败（重试{max_retries}次后）: {last_error}"
    }
