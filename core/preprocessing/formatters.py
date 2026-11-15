"""
格式化工具函数

提取自原始预处理脚本的格式化逻辑，保持完全一致以确保输出一致性。
"""

from typing import Dict, Any, List


def format_tool_parameters(parameters: Dict[str, Any]) -> str:
    """
    将工具参数信息格式化为文本

    来源：build_faiss_tools_base_model_with_parameters.py

    Args:
        parameters: 参数字典，包含 'basic' 和 'advanced' 两个部分

    Returns:
        格式化的参数文本
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
    将工具信息格式化为文本（带参数信息）
    与向量数据库中的格式完全一致

    这个函数必须与原代码完全一致，因为输出会影响：
    1. 三元组中的工具文本表示
    2. CSV转换时的ID映射
    3. 最终训练数据的格式

    Args:
        tool: 工具信息字典，包含以下字段：
            - name: 工具名称
            - algorithm_id: 算法ID
            - description: 工具描述
            - parameters: 参数字典

    Returns:
        格式化的工具文本字符串

    示例输出:
        Tool: Buffer
        Algorithm ID: native:buffer
        Description: Creates buffer polygons around features
        Parameters:
        - distance (Number): Buffer distance
        - segments (Integer): Number of segments
    """
    tool_text_parts = [
        f"Tool: {tool.get('name', 'Unknown')}",
        f"Algorithm ID: {tool.get('algorithm_id', 'unknown')}",
        f"Description: {tool.get('description', '')}"
    ]

    # 添加参数信息
    parameters = tool.get('parameters', {})
    param_text = format_tool_parameters(parameters)
    if param_text:
        tool_text_parts.append(f"Parameters:\n{param_text}")

    return "\n".join(tool_text_parts)


def format_toolbox_info(toolbox: Dict[str, Any]) -> str:
    """
    将工具箱信息格式化为文本

    Args:
        toolbox: 工具箱信息字典，包含以下字段：
            - toolbox_name: 工具箱名称
            - toolbox_description: 工具箱描述

    Returns:
        格式化的工具箱文本字符串

    示例输出:
        Toolbox: Vector geometry
        Description: Tools for vector geometry operations
    """
    return f"Toolbox: {toolbox.get('toolbox_name', 'Unknown')}\nDescription: {toolbox.get('toolbox_description', '')}"


def extract_algorithm_id_from_tool_text(tool_text: str) -> str:
    """
    从格式化的工具文本中提取algorithm_id

    Args:
        tool_text: 格式化的工具文本（由format_tool_info生成）

    Returns:
        提取的algorithm_id字符串，如果未找到则返回空字符串

    示例:
        输入: "Tool: Buffer\nAlgorithm ID: native:buffer\n..."
        输出: "native:buffer"
    """
    for line in tool_text.split('\n'):
        if line.startswith('Algorithm ID:'):
            return line.replace('Algorithm ID:', '').strip()
    return ""


def extract_algorithm_ids(tool_texts: List[str]) -> set:
    """
    从工具文本列表中提取所有的algorithm_id

    Args:
        tool_texts: 格式化的工具文本列表

    Returns:
        algorithm_id的集合
    """
    algorithm_ids = set()
    for text in tool_texts:
        algorithm_id = extract_algorithm_id_from_tool_text(text)
        if algorithm_id:
            algorithm_ids.add(algorithm_id)
    return algorithm_ids


def extract_toolbox_name(toolbox: Dict[str, Any]) -> str:
    """
    从工具箱字典中提取工具箱名称

    Args:
        toolbox: 工具箱字典

    Returns:
        工具箱名称字符串
    """
    return toolbox.get('toolbox_name', '')


def extract_toolbox_names(toolbox_list: List[Dict[str, Any]]) -> set:
    """
    从工具箱字典列表中提取所有的toolbox_name

    Args:
        toolbox_list: 工具箱字典列表

    Returns:
        toolbox_name的集合
    """
    toolbox_names = set()
    for toolbox in toolbox_list:
        if isinstance(toolbox, dict):
            toolbox_name = extract_toolbox_name(toolbox)
            if toolbox_name:
                toolbox_names.add(toolbox_name)
    return toolbox_names


def load_all_tools(tools_file: str) -> List[Dict[str, Any]]:
    """
    从工具文件中加载所有工具

    Args:
        tools_file: 工具数据文件路径（JSON格式）

    Returns:
        所有工具的列表
    """
    import json

    with open(tools_file, 'r', encoding='utf-8') as f:
        toolboxes = json.load(f)

    all_tools = []
    for toolbox in toolboxes:
        for tool in toolbox.get('tools', []):
            all_tools.append(tool)

    return all_tools


def load_all_toolboxes(tools_file: str) -> List[Dict[str, Any]]:
    """
    从工具文件中加载所有工具箱

    Args:
        tools_file: 工具数据文件路径（JSON格式）

    Returns:
        所有工具箱的列表
    """
    import json

    with open(tools_file, 'r', encoding='utf-8') as f:
        toolboxes_data = json.load(f)

    all_toolboxes = []
    for toolbox in toolboxes_data:
        toolbox_info = {
            'toolbox_name': toolbox.get('toolbox', ''),
            'toolbox_description': toolbox.get('discription', '')  # 注意：原文件使用 'discription'（拼写错误）
        }
        all_toolboxes.append(toolbox_info)

    return all_toolboxes
