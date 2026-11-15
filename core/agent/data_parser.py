#!/usr/bin/env python3
"""
统一的数据解析模块
保证训练数据、测试集、向量库使用相同的解析格式
"""

import json
from typing import Dict, List, Any, Tuple


def format_tool_parameters(parameters: Dict[str, Any]) -> str:
    """
    将参数信息格式化为文本

    Args:
        parameters: 参数字典，包含 'basic' 和 'advanced' 两个键

    Returns:
        格式化后的参数文本
    """
    if not parameters:
        return ""

    param_texts = []

    # 处理基本参数
    basic_params = parameters.get('basic', {})
    if basic_params:
        param_texts.append("Basic Parameters:")
        for param_name, param_info in basic_params.items():
            param_desc = param_info.get('description', '')
            param_type = param_info.get('type', '')
            default_val = param_info.get('default', '')

            param_text = f"  - {param_name}: {param_desc}"
            if param_type:
                param_text += f" (Type: {param_type})"
            if default_val:
                param_text += f" [Default: {default_val}]"

            param_texts.append(param_text)

    # 处理高级参数
    advanced_params = parameters.get('advanced', {})
    if advanced_params:
        param_texts.append("Advanced Parameters:")
        for param_name, param_info in advanced_params.items():
            param_desc = param_info.get('description', '')
            param_type = param_info.get('type', '')
            default_val = param_info.get('default', '')

            param_text = f"  - {param_name}: {param_desc}"
            if param_type:
                param_text += f" (Type: {param_type})"
            if default_val:
                param_text += f" [Default: {default_val}]"

            param_texts.append(param_text)

    return "\n".join(param_texts)


def format_tool_info(tool: Dict[str, Any]) -> str:
    """
    格式化工具信息为标准字符串

    Args:
        tool: 工具信息字典

    Returns:
        格式化后的工具信息字符串
    """
    tool_name = tool.get('name', '')
    tool_desc = tool.get('description', '')
    algorithm_id = tool.get('algorithm_id', '')
    parameters = tool.get('parameters', {})

    # 构建工具文本
    tool_text_parts = [
        f"Tool: {tool_name}",
        f"Algorithm ID: {algorithm_id}",
        f"Description: {tool_desc}"
    ]

    # 添加参数信息
    param_text = format_tool_parameters(parameters)
    if param_text:
        tool_text_parts.append(f"Parameters:\n{param_text}")

    return "\n".join(tool_text_parts)


def format_toolbox_info(toolbox: Dict[str, Any]) -> str:
    """
    格式化工具箱信息为标准字符串

    Args:
        toolbox: 工具箱信息字典

    Returns:
        格式化后的工具箱信息字符串
    """
    toolbox_name = toolbox.get('name', '')
    toolbox_desc = toolbox.get('description', '')

    return f"Toolbox: {toolbox_name}\nDescription: {toolbox_desc}"


def extract_tools_and_toolboxes(data: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    从原始数据提取并格式化工具和工具箱信息

    Args:
        data: 工具箱列表，每个工具箱包含多个工具

    Returns:
        包含以下键的字典:
        - tools: 格式化后的工具文本列表
        - toolboxes: 格式化后的工具箱文本列表
        - tools_metadata: 工具元数据列表
        - toolboxes_metadata: 工具箱元数据列表
    """
    tools_texts = []
    toolboxes_texts = []
    tools_metadata = []
    toolboxes_metadata = []

    for toolbox_idx, toolbox in enumerate(data):
        if not isinstance(toolbox, dict):
            continue

        toolbox_name = toolbox.get('toolbox', '')
        toolbox_desc = toolbox.get('discription', '')
        tools = toolbox.get('tools', [])

        # 处理工具箱信息
        if toolbox_name.strip():
            toolbox_text = f"Toolbox: {toolbox_name}\nDescription: {toolbox_desc}"
            toolboxes_texts.append(toolbox_text)
            toolboxes_metadata.append({
                "type": "toolbox",
                "name": toolbox_name,
                "description": toolbox_desc,
                "toolbox_index": toolbox_idx
            })

        # 处理每个工具
        for tool_idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                continue

            tool_name = tool.get('name', '')
            tool_desc = tool.get('description', '')
            algorithm_id = tool.get('algorithm_id', '')
            parameters = tool.get('parameters', {})

            if not tool_name.strip():
                continue

            # 使用统一的格式化函数
            tool_text = format_tool_info(tool)
            tools_texts.append(tool_text)

            # 构建元数据
            tools_metadata.append({
                "type": "tool",
                "name": tool_name,
                "algorithm_id": algorithm_id,
                "description": tool_desc,
                "toolbox": toolbox_name,
                "toolbox_index": toolbox_idx,
                "tool_index": tool_idx,
                "has_parameters": bool(parameters),
                "parameters": parameters
            })

    return {
        "tools": tools_texts,
        "toolboxes": toolboxes_texts,
        "tools_metadata": tools_metadata,
        "toolboxes_metadata": toolboxes_metadata
    }


def build_algorithm_to_toolbox_mapping(tool_data_file: str) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    构建 algorithm_id 到工具箱的映射关系

    Args:
        tool_data_file: 工具数据JSON文件路径（qgis_tool_with_paramaters重写工具集描述.json）

    Returns:
        (algorithm_to_toolbox_dict, algorithm_to_toolbox_string):
        - algorithm_to_toolbox_dict: {algorithm_id: toolbox_dict}
        - algorithm_to_toolbox_string: {algorithm_id: formatted_toolbox_string}
    """
    with open(tool_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    algorithm_to_toolbox_dict = {}
    algorithm_to_toolbox_string = {}

    for toolbox in data:
        toolbox_name = toolbox.get('toolbox', '')
        toolbox_desc = toolbox.get('discription', '')

        toolbox_dict = {
            'name': toolbox_name,
            'description': toolbox_desc
        }

        toolbox_string = format_toolbox_info(toolbox_dict)

        # 遍历该工具箱下的所有工具
        for tool in toolbox.get('tools', []):
            algorithm_id = tool.get('algorithm_id', '')
            if algorithm_id:
                algorithm_to_toolbox_dict[algorithm_id] = toolbox_dict
                algorithm_to_toolbox_string[algorithm_id] = toolbox_string

    return algorithm_to_toolbox_dict, algorithm_to_toolbox_string


def format_case_description(case_desc: str) -> str:
    """
    格式化案例描述

    Args:
        case_desc: 案例描述文本

    Returns:
        格式化后的案例描述（目前直接返回，保持一致性）
    """
    # 案例描述直接使用原文
    return case_desc


def extract_cases_from_test_file(test_file: str) -> Dict[str, List]:
    """
    从反向测试集文件中提取案例描述
    格式: {工具描述: [案例1, 案例2, ...]}

    Args:
        test_file: 反向测试集文件路径 (case_from_tool_*.json)

    Returns:
        包含以下键的字典:
        - cases: 案例描述列表（去重）
        - cases_metadata: 案例元数据列表
    """
    with open(test_file, 'r', encoding='utf-8') as f:
        tool_to_cases = json.load(f)

    # 收集所有唯一的案例描述
    unique_cases = set()
    for cases in tool_to_cases.values():
        unique_cases.update(cases)

    # 转换为列表并格式化
    cases_texts = []
    cases_metadata = []

    for case_idx, case_desc in enumerate(sorted(unique_cases)):
        # 格式化案例描述
        formatted_case = format_case_description(case_desc)
        cases_texts.append(formatted_case)

        # 构建元数据
        cases_metadata.append({
            "type": "case",
            "case_index": case_idx,
            "description": case_desc,
            "length": len(case_desc)
        })

    return {
        "cases": cases_texts,
        "cases_metadata": cases_metadata
    }
