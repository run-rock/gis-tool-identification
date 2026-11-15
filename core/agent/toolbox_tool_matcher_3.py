#!/usr/bin/env python3
"""
工具识别器 v3

简化的单次识别逻辑：
1. 检索 k 个工具箱（仅用于后续验证，不做LLM识别）
2. 检索 l 个工具
3. LLM只对工具进行识别（一次性，不迭代）
4. 判断选中工具所属工具箱是否在检索的k个工具箱中
5. 不迭代，直接返回结果
"""

import os
from typing import Dict, List, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class ToolSelection(BaseModel):
    """工具选择结构化输出（多工具场景）"""
    selected_tool_index: int = Field(
        description="Index of the selected tool (starting from 0)",
        ge=0
    )


class SingleToolValidation(BaseModel):
    """单工具验证输出（允许返回-1表示工具不可用）"""
    selected_tool_index: int = Field(
        description="Index of the tool if applicable (0 if can solve, -1 if cannot solve)",
        ge=-1  # 允许-1表示工具不可用
    )


class ToolMatcher:
    """工具识别器（简化版，无迭代）"""

    def __init__(self,
                 toolbox_tool_mapping: Dict[str, List[str]],
                 llm_provider: str = "qwen3",
                 llm_model: str = "qwen3-32b-gguf",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 use_no_think: bool = False,
                 temperature: float = 0.3,
                 verbose: bool = False):
        """
        初始化

        Args:
            toolbox_tool_mapping: 工具箱到工具的映射 {toolbox_text: [tool_algorithm_ids]}
            llm_provider: LLM供应商 ("qwen3", "openai", "deepseek" 等)
            llm_model: 模型名称
            api_key: API密钥（可选，默认从环境变量读取）
            base_url: API基础URL（可选，默认从环境变量读取）
            use_no_think: 是否使用/no_think（Qwen3）
            temperature: 温度参数
            verbose: 是否打印详细信息
        """
        # 加载环境变量
        load_dotenv()

        # 根据provider设置API配置
        if api_key is None or base_url is None:
            if llm_provider == "qwen3":
                api_key = api_key or os.getenv("QWEN3_KEY")
                base_url = base_url or os.getenv("QWEN3_URL")
            elif llm_provider == "YINLI_AI":
                api_key = api_key or os.getenv("YINLI_AI_KEY")
                base_url = base_url or os.getenv("YINLI_AI_URL")
            elif llm_provider == "moonshot":
                api_key = api_key or os.getenv("FENG_KIMI_API_KEY")
                base_url = base_url or os.getenv("FENG_KIMI_URL")
            else:
                raise ValueError(f"不支持的LLM供应商: {llm_provider}，支持的供应商: qwen3, YINLI_AI, moonshot")

        if not api_key:
            raise ValueError(f"未找到API密钥，请设置环境变量 {llm_provider.upper()}_KEY 或手动提供api_key参数")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_provider = llm_provider
        self.model_name = llm_model
        self.toolbox_tool_mapping = toolbox_tool_mapping
        self.use_no_think = use_no_think
        self.temperature = temperature
        self.verbose = verbose

        if self.verbose:
            print(f"✓ LLM配置: {llm_provider} / {llm_model}")
            print(f"✓ Base URL: {base_url}")

    def _build_tool_selector_prompt(self,
                                     query: str,
                                     candidate_tools: List[Dict[str, Any]]) -> str:
        """构建工具选择器提示词"""
        # 构建工具候选列表
        tools_text = ""
        for i, tool in enumerate(candidate_tools):
            tools_text += f"\n[Tool {i}]\n"
            tools_text += tool['tool_text']
            tools_text += "\n"

        if len(candidate_tools) >= 1:
            prompt = f"""You are a QGIS tool selection expert. Please select the tool that can accomplish the user's task.

            User Task:
            {query}

            Candidate Tools:
            {tools_text}


            Please analyze:
            1. What type of data does the user need to process?
            2. What output does the user want to get?
            3. Which tool can accomplish this task?

            Return:
            - selected_index: Index of the selected tool (0 to {len(candidate_tools)-1})"""
        
        if len(candidate_tools) == 1:
            prompt = f"""You are a QGIS tool selection expert. Please determine whether the tool can accomplish the user's task.

            User Task:
            {query}

            Candidate Tool:
            {tools_text}


            Please analyze:
            1. What type of data does the user need to process?
            2. What output does the user want to get?
            3. Can this tool accomplish the user's task?

            Return:
            - selected_tool_index:
              * Return 0 if the tool CAN accomplish the task
              * Return -1 if the tool CANNOT accomplish the task"""

        return prompt

    def _call_tool_selector(self,
                           query: str,
                           candidate_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用工具选择器"""
        prompt = self._build_tool_selector_prompt(query, candidate_tools)

        user_content = prompt
        if self.use_no_think:
            user_content = prompt + " /no_think"

        if self.verbose:
            print("\n" + "=" * 80)
            print("📥 工具选择器输入:")
            print("=" * 80)
            print(user_content)
            print("=" * 80)

        try:
            # 根据候选工具数量选择不同的响应格式
            response_format = SingleToolValidation if len(candidate_tools) == 1 else ToolSelection

            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": user_content}],
                response_format=response_format,
                temperature=self.temperature,
            )

            result = response.choices[0].message.parsed

            if self.verbose:
                print("\n" + "=" * 80)
                print("🔧 工具选择器输出:")
                print("=" * 80)
                print(f"  Tool Index: {result.selected_tool_index}")
                print("=" * 80)

            # 获取工具索引
            tool_idx = result.selected_tool_index

            # 单工具场景：处理 -1（工具不可用）
            if len(candidate_tools) == 1 and tool_idx == -1:
                return {
                    "success": True,
                    "selected_tool_index": -1,
                    "selected_tool": None,
                    "is_tool_invalid": True  # 标记为工具不可用
                }

            # 多工具场景或单工具返回0：验证索引范围并修正
            if tool_idx < 0 or tool_idx >= len(candidate_tools):
                tool_idx = max(0, min(tool_idx, len(candidate_tools) - 1))

            return {
                "success": True,
                "selected_tool_index": tool_idx,
                "selected_tool": candidate_tools[tool_idx],
                "is_tool_invalid": False
            }

        except Exception as e:
            import traceback
            error_msg = f"Tool selector call failed: {e}\n{traceback.format_exc()}"
            print(f"❌ 工具选择器调用失败:")
            print(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    def _find_tool_toolbox(self, tool_algorithm_id: str) -> str:
        """找到工具所属的工具箱文本"""
        for toolbox_text, tools in self.toolbox_tool_mapping.items():
            if tool_algorithm_id in tools:
                return toolbox_text
        return ""

    def identify(self,
                query: str,
                candidate_toolboxes: List[Dict[str, Any]],
                candidate_tools: List[Dict[str, Any]],
                extract_tool_algorithm_id_fn) -> Dict[str, Any]:
        """
        执行工具识别（单次，无迭代）

        Args:
            query: 用户查询
            candidate_toolboxes: 候选工具箱列表 [{'toolbox_text': '...', ...}, ...]
            candidate_tools: 候选工具列表 [{'tool_text': '...', ...}, ...]
            extract_tool_algorithm_id_fn: 从工具文本提取算法ID的函数

        Returns:
            识别结果
        """
        if not candidate_tools:
            return {
                "success": False,
                "error": "Empty candidate tools"
            }

        if self.verbose:
            print("\n" + "=" * 80)
            print("📝 用户查询:")
            print("=" * 80)
            print(query)
            print("=" * 80)

        print(f"\n🔄 开始工具识别...")

        # 调用工具选择器
        print("🎯 选择工具...")
        selection_result = self._call_tool_selector(query, candidate_tools)

        if not selection_result["success"]:
            return {
                "success": False,
                "error": selection_result["error"]
            }

        # 获取选择结果
        selected_tool_index = selection_result["selected_tool_index"]
        is_tool_invalid = selection_result.get("is_tool_invalid", False)

        # 单工具场景：如果返回-1，视为不合理识别
        if is_tool_invalid:
            print(f"   ❌ 工具被判定为不可用（返回-1）")
            print(f"🔍 验证结果：不合理！工具不适用于该任务")

            return {
                "success": True,
                "selected_tool_index": -1,
                "selected_tool_algorithm_id": None,
                "tool_belongs_to_toolbox": None,
                "is_reasonable": False,  # 视为不合理识别
                "reason": "tool_invalid"  # 标记原因
            }

        # 正常情况：工具有效
        selected_tool = selection_result["selected_tool"]
        tool_algorithm_id = extract_tool_algorithm_id_fn(selected_tool['tool_text'])

        print(f"   选择工具 {selected_tool_index}: {tool_algorithm_id}")

        # 找到工具所属的工具箱
        tool_belongs_to_toolbox_text = self._find_tool_toolbox(tool_algorithm_id)

        # 提取候选工具箱的文本列表
        candidate_toolbox_texts = [tb['toolbox_text'] for tb in candidate_toolboxes]

        # 判断工具所属工具箱是否在候选工具箱中
        is_reasonable = tool_belongs_to_toolbox_text in candidate_toolbox_texts

        print(f"🔍 验证：工具所属工具箱是否在候选中...")
        if is_reasonable:
            print("✅ 合理！工具所属工具箱在检索结果中")
        else:
            print("❌ 不合理！工具所属工具箱不在检索结果中")
            if tool_belongs_to_toolbox_text:
                print(f"   工具实际属于: {tool_belongs_to_toolbox_text[:50]}...")
            else:
                print(f"   工具 {tool_algorithm_id} 未找到所属工具箱")

        return {
            "success": True,
            "selected_tool_index": selected_tool_index,
            "selected_tool_algorithm_id": tool_algorithm_id,
            "tool_belongs_to_toolbox": tool_belongs_to_toolbox_text,
            "is_reasonable": is_reasonable,
            "reason": "toolbox_match" if is_reasonable else "toolbox_mismatch"
        }
