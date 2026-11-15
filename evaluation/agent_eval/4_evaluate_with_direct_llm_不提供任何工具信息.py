#!/usr/bin/env python3
"""
直接LLM识别评估：不提供任何工具信息，让LLM直接推荐工具

使用示例：
nohup python -u embedding_model_train/toolbench/evaluate/G8/scripts/4_evaluate_with_direct_llm_不提供任何工具信息.py \
    --test_file data/retrival_G8/测试数据/工具检索和识别/v1-6-人工校验v2.json \
    --output_file casesynthesis_qgis/data/retrival_G8/测试结果-工具识别/direct_llm不提供工具信息/kimi.json \
    --llm_provider kimi \
    --llm_model kimi-k2-0905-preview \
    --max_workers 1 > casesynthesis_qgis/data/retrival_G8/测试结果-工具识别/direct_llm不提供工具信息/kimi.log 2>&1 &
"""

import json
import sys
import os
import argparse
import threading
import re
import time
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pydantic import BaseModel, Field

# 添加core目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ToolIdentification(BaseModel):
    """工具识别结果的结构化输出"""
    tool_name: str = Field(
        description="The name of the recommended QGIS tool"
    )


def load_test_data(test_file: str) -> Dict[str, List[str]]:
    """加载测试数据"""
    print(f"📂 加载测试数据: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   测试案例数量: {len(data)}")
    return data


def build_prompt(case_desc: str) -> str:
    """构建提示词"""
    prompt = f"""You are a QGIS tool selection expert. Based on the given user task description, recommend a QGIS tool for the user. You only need to output the tool name in English.

User Task:
{case_desc}


Please return in the following format:
Recommended Tool Name: buffer
"""

    return prompt


def call_llm(client: OpenAI, prompt: str, model_name: str, use_no_think: bool = False, max_retries: int = 3) -> Dict[str, Any]:
    """调用LLM（使用结构化输出）"""
    # 如果是Qwen3，在prompt后添加 /no_think
    user_content = prompt
    if use_no_think:
        user_content = prompt + " /no_think"

    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                response_format=ToolIdentification
            )

            # 获取结构化输出
            tool_identification = response.choices[0].message.parsed

            if tool_identification and tool_identification.tool_name:
                return {
                    "success": True,
                    "tool_name": tool_identification.tool_name
                }
            else:
                # 如果解析失败，尝试从原始文本提取
                result_text = response.choices[0].message.content
                return {
                    "success": True,
                    "result_text": result_text,
                    "tool_name": None
                }

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️  LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                continue
            else:
                return {
                    "success": False,
                    "error": str(e)
                }


def evaluate_single_case(case_desc: str,
                         ground_truth_tools: List[str],
                         llm_client: OpenAI,
                         llm_model: str,
                         use_no_think: bool = False) -> Dict[str, Any]:
    """评估单个案例"""
    start_time = time.time()

    result = {
        "case": case_desc,
        "ground_truth": ground_truth_tools,
        "success": False,
        "error_type": None,
        "error_message": None,
        "selected_tool_name": None,
        "is_correct": False,
        "processing_time": 0.0
    }

    try:
        # 1. 构建prompt
        prompt = build_prompt(case_desc)

        # 2. 调用LLM
        llm_result = call_llm(llm_client, prompt, llm_model, use_no_think)

        if not llm_result["success"]:
            result["error_type"] = "llm_failed"
            result["error_message"] = llm_result.get("error", "LLM调用失败")
            return result

        # 3. 获取工具名称（使用结构化输出）
        selected_name = llm_result.get("tool_name")

        if not selected_name:
            result["error_type"] = "no_tool_name_found"
            result["error_message"] = "LLM未返回工具名称"
            return result

        result["selected_tool_name"] = selected_name
        result["success"] = True

        # 4. 从ground_truth中提取工具名称
        ground_truth_names = []
        for gt_tool in ground_truth_tools:
            # 提取 "Tool: xxx" 中的工具名称
            match = re.search(r'Tool:\s*([^\n]+)', gt_tool)
            if match:
                tool_name = match.group(1).strip()
                ground_truth_names.append(tool_name)

        # 5. 判断是否正确（按工具名称匹配，不区分大小写）
        selected_name_lower = selected_name.lower()
        ground_truth_names_lower = [name.lower() for name in ground_truth_names]
        result["is_correct"] = selected_name_lower in ground_truth_names_lower
        result["ground_truth_names"] = ground_truth_names

    except Exception as e:
        result["error_type"] = "exception"
        result["error_message"] = str(e)

    # 记录处理时间
    end_time = time.time()
    result["processing_time"] = end_time - start_time

    return result


def main():
    parser = argparse.ArgumentParser(description="直接LLM识别评估（无检索）")

    # 必需参数
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                        help="评估结果输出文件路径")

    # LLM参数
    parser.add_argument("--llm_provider", type=str, default="kimi",
                        choices=["kimi", "qwen3"],
                        help="LLM提供商（kimi或qwen3）")
    parser.add_argument("--llm_model", type=str, default="kimi-k2-0905-preview",
                        help="LLM模型名称")

    # 并行处理参数
    parser.add_argument("--max_workers", type=int, default=10,
                        help="最大并行线程数")

    args = parser.parse_args()

    # 加载环境变量
    load_dotenv('.env')

    # 根据provider选择API配置
    if args.llm_provider == "qwen3":
        llm_api_key = os.getenv('QWEN3_KEY')
        llm_base_url = os.getenv('QWEN3_URL')
    else:  # kimi
        llm_api_key = os.getenv('FENG_KIMI_API_KEY')
        llm_base_url = os.getenv('FENG_KIMI_URL')

    if not llm_api_key or not llm_base_url:
        raise ValueError("未找到LLM API配置")

    print("=" * 80)
    print("直接LLM识别评估（不提供工具信息）")
    print("=" * 80)
    print(f"📍 LLM提供商: {args.llm_provider}")
    print(f"🤖 LLM模型: {args.llm_model}")
    print(f"🔧 使用 /no_think: {args.llm_provider == 'qwen3'}")

    # 1. 加载测试数据
    test_data = load_test_data(args.test_file)

    # 2. 创建LLM客户端
    llm_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)

    # 5. 多线程评估
    print("\n" + "=" * 80)
    print("🚀 开始多线程评估...")
    print(f"🧵 并行线程数: {args.max_workers}")
    print("=" * 80)

    # 记录整体开始时间（墙钟时间）
    overall_start_time = time.time()

    all_results = []
    error_stats = {
        "llm_failed": 0,
        "no_tool_name_found": 0,
        "exception": 0
    }
    error_cases = {
        "llm_failed": [],
        "no_tool_name_found": [],
        "exception": [],
        "wrong_prediction": []
    }
    correct_count = 0
    total_count = len(test_data)
    processing_times = []  # 记录每个案例的处理时间

    # 线程锁
    results_lock = threading.Lock()
    stats_lock = threading.Lock()

    def process_case_wrapper(idx_case_gt: Tuple[int, str, List[str]]) -> Dict[str, Any]:
        """处理单个案例"""
        idx, case_desc, ground_truth = idx_case_gt

        print(f"\n[{idx}/{total_count}] 评估案例...")
        print(f"   案例: {case_desc[:100]}...")

        result = evaluate_single_case(
            case_desc=case_desc,
            ground_truth_tools=ground_truth,
            llm_client=llm_client,
            llm_model=args.llm_model,
            use_no_think=(args.llm_provider == "qwen3")
        )

        # 线程安全地更新统计
        nonlocal correct_count
        with stats_lock:
            # 记录处理时间
            processing_times.append(result["processing_time"])
            print(f"   ⏱️  处理时间: {result['processing_time']:.2f}秒")

            if result["success"]:
                if result["is_correct"]:
                    correct_count += 1
                    print(f"   ✅ 正确")
                    print(f"   选中工具: {result['selected_tool_name']}")
                else:
                    # 大模型识别错了
                    error_cases["wrong_prediction"].append({
                        "case": case_desc,
                        "ground_truth_names": result.get("ground_truth_names", []),
                        "selected_tool_name": result["selected_tool_name"]
                    })
                    print(f"   ❌ 错误")
                    print(f"   正确工具: {result.get('ground_truth_names', [])}")
                    print(f"   选中工具: {result['selected_tool_name']}")
            else:
                error_type = result["error_type"]
                error_stats[error_type] += 1
                error_cases[error_type].append({
                    "case": case_desc,
                    "ground_truth_names": result.get("ground_truth_names", []),
                    "error_message": result["error_message"]
                })
                print(f"   ❌ 失败: {error_type}")

        return result

    # 准备任务列表
    tasks = [(idx, case_desc, ground_truth)
             for idx, (case_desc, ground_truth) in enumerate(test_data.items(), 1)]

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_case_wrapper, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                with results_lock:
                    all_results.append(result)
            except Exception as e:
                task = futures[future]
                print(f"\n❌ 任务 {task[0]} 执行异常: {e}")

    # 6. 统计结果
    print("\n" + "=" * 80)
    # 记录整体结束时间（墙钟时间）
    overall_end_time = time.time()
    overall_processing_time = overall_end_time - overall_start_time

    print("📊 评估结果统计")
    print("=" * 80)
    print(f"总案例数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {correct_count / total_count * 100:.2f}%")

    # 处理时间统计
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        sum_time = sum(processing_times)
        print(f"\n⏱️  处理时间统计:")
        print(f"  平均时间: {avg_time:.2f}秒/案例")
        print(f"  最快时间: {min_time:.2f}秒")
        print(f"  最慢时间: {max_time:.2f}秒")
        print(f"  总执行时间（墙钟时间）: {overall_processing_time:.2f}秒")
        print(f"  各案例时间之和: {sum_time:.2f}秒")
        parallel_efficiency = sum_time / overall_processing_time if overall_processing_time > 0 else 0
        efficiency_percentage = (parallel_efficiency / args.max_workers) * 100 if args.max_workers > 0 else 0
        print(f"  并行效率: {parallel_efficiency:.2f}x (理论最大={args.max_workers}x, 达到{efficiency_percentage:.1f}%)")

    print("\n错误类型统计:")
    for error_type, count in error_stats.items():
        if count > 0:
            print(f"  - {error_type}: {count}")

    wrong_prediction_count = len(error_cases["wrong_prediction"])
    if wrong_prediction_count > 0:
        print(f"  - wrong_prediction: {wrong_prediction_count}")

    # 7. 保存结果
    print("\n" + "=" * 80)
    print(f"💾 保存结果到: {args.output_file}")

    # 创建输出目录
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 计算时间统计
    time_stats = {}
    if processing_times:
        sum_time = sum(processing_times)
        parallel_efficiency = sum_time / overall_processing_time if overall_processing_time > 0 else 0
        efficiency_percentage = (parallel_efficiency / args.max_workers) * 100 if args.max_workers > 0 else 0

        time_stats = {
            "average_time_per_case": sum_time / len(processing_times),
            "min_time": min(processing_times),
            "max_time": max(processing_times),
            "sum_of_case_times": sum_time,
            "overall_processing_time": overall_processing_time,
            "parallel_efficiency": parallel_efficiency,
            "efficiency_percentage": efficiency_percentage,
            "max_workers": args.max_workers
        }

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "test_file": args.test_file,
            "llm_model": args.llm_model
        },
        "summary": {
            "total_cases": total_count,
            "correct_count": correct_count,
            "accuracy": correct_count / total_count,
            "error_stats": {
                **error_stats,
                "wrong_prediction": wrong_prediction_count
            },
            "processing_time_stats": time_stats
        },
        "error_cases": {
            "llm_failed": {
                "description": "大模型调用失败",
                "count": len(error_cases["llm_failed"]),
                "cases": error_cases["llm_failed"]
            },
            "no_tool_name_found": {
                "description": "LLM响应中未找到工具名称",
                "count": len(error_cases["no_tool_name_found"]),
                "cases": error_cases["no_tool_name_found"]
            },
            "exception": {
                "description": "程序运行异常",
                "count": len(error_cases["exception"]),
                "cases": error_cases["exception"]
            },
            "wrong_prediction": {
                "description": "大模型识别错了",
                "count": len(error_cases["wrong_prediction"]),
                "cases": error_cases["wrong_prediction"]
            }
        }
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 结果已保存")
    print("=" * 80)


if __name__ == "__main__":
    main()
