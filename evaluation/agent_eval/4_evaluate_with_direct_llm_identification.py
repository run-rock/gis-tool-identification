#!/usr/bin/env python3
"""
直接LLM识别评估：不经过检索，直接将所有工具信息提供给LLM进行识别

使用示例：
nohup python embedding_model_train/toolbench/evaluate/G8/scripts/4_evaluate_with_direct_llm_identification.py \
    --test_file data/retrival_G8/测试数据/工具检索和识别/v1-6-人工校验v2.json \
    --tools_file casesynthesis_qgis/data/qgis_tool/all_tools_without_params.txt \
    --output_file casesynthesis_qgis/data/retrival_G8/测试结果-工具识别/direct_llm/kimi-result.json \
    --llm_provider kimi \
    --llm_model kimi-k2-0905-preview \
    --max_workers 1 > casesynthesis_qgis/data/retrival_G8/测试结果-工具识别/direct_llm/kimi-result1.log 2>&1 &
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
    algorithm_id: str = Field(
        description="选中的工具的Algorithm ID"
    )


def load_test_data(test_file: str) -> Dict[str, List[str]]:
    """加载测试数据"""
    print(f"📂 加载测试数据: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   测试案例数量: {len(data)}")
    return data


def load_all_tools_text(tools_file: str) -> str:
    """加载所有工具文本（不带参数）"""
    print(f"📂 加载所有工具文本: {tools_file}")
    with open(tools_file, 'r', encoding='utf-8') as f:
        all_tools_text = f.read()
    print(f"   文本长度: {len(all_tools_text)} 字符")
    return all_tools_text


def extract_algorithm_ids_from_tools_text(tools_text: str) -> List[str]:
    """从工具文本中提取所有算法ID"""
    pattern = r'Algorithm ID: ([^\n]+)'
    algorithm_ids = re.findall(pattern, tools_text)
    return algorithm_ids


def build_prompt(case_desc: str, all_tools_text: str) -> str:
    """构建提示词"""
    prompt = f"""你是一个QGIS工具选择专家。给定用户的任务描述和所有可用的QGIS工具，请选择最适合完成该任务的工具。

用户任务:
{case_desc}

所有可用工具:
{all_tools_text}

用户任务:
{case_desc}

请仔细分析用户的任务需求，从上述所有工具中选择**最合适**的工具。

**重要说明**:
1. 每个工具都有一个Algorithm ID（格式如：gdal:aspect, native:buffer等）
2. 你需要返回选中工具的完整Algorithm ID
3. 只返回Algorithm ID，不要返回其他内容

请按照以下格式返回:
Algorithm ID: <你选择的工具的算法ID>

例如:
Algorithm ID: gdal:aspect
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

            if tool_identification and tool_identification.algorithm_id:
                return {
                    "success": True,
                    "algorithm_id": tool_identification.algorithm_id
                }
            else:
                # 如果解析失败，尝试从原始文本提取
                result_text = response.choices[0].message.content
                return {
                    "success": True,
                    "result_text": result_text,
                    "algorithm_id": None
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


def extract_algorithm_ids_from_response(response_text: str) -> List[str]:
    """从LLM响应中提取算法ID"""
    # 查找 "Algorithm ID:" 后面的内容
    pattern = r'Algorithm ID:\s*([^\n]+)'
    match = re.search(pattern, response_text, re.IGNORECASE)

    if match:
        ids_text = match.group(1).strip()
        # 分割多个ID（用逗号分隔）
        algorithm_ids = [id.strip() for id in ids_text.split(',')]
        return algorithm_ids
    else:
        return []


def evaluate_single_case(case_desc: str,
                         ground_truth_tools: List[str],
                         all_tools_text: str,
                         valid_algorithm_ids: List[str],
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
        "selected_algorithm_id": None,
        "is_correct": False,
        "processing_time": 0.0
    }

    try:
        # 1. 构建prompt
        prompt = build_prompt(case_desc, all_tools_text)

        # 2. 调用LLM
        llm_result = call_llm(llm_client, prompt, llm_model, use_no_think)

        if not llm_result["success"]:
            result["error_type"] = "llm_failed"
            result["error_message"] = llm_result.get("error", "LLM调用失败")
            return result

        # 3. 获取算法ID（使用结构化输出）
        selected_id = llm_result.get("algorithm_id")

        if not selected_id:
            result["error_type"] = "no_algorithm_id_found"
            result["error_message"] = "LLM未返回Algorithm ID"
            return result

        result["selected_algorithm_id"] = selected_id
        result["success"] = True

        # 4. 从ground_truth中提取算法ID
        ground_truth_ids = []
        for gt_tool in ground_truth_tools:
            match = re.search(r'Algorithm ID: ([^\n]+)', gt_tool)
            if match:
                ground_truth_ids.append(match.group(1).strip())

        # 5. 判断是否正确（选中的ID在ground_truth中就算对）
        result["is_correct"] = selected_id in ground_truth_ids
        result["ground_truth_ids"] = ground_truth_ids

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
    parser.add_argument("--tools_file", type=str, required=True,
                        help="所有工具文本文件路径（不带参数）")
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
    print("直接LLM识别评估（无检索）")
    print("=" * 80)
    print(f"📍 LLM提供商: {args.llm_provider}")
    print(f"🤖 LLM模型: {args.llm_model}")
    print(f"🔧 使用 /no_think: {args.llm_provider == 'qwen3'}")

    # 1. 加载测试数据
    test_data = load_test_data(args.test_file)

    # 2. 加载所有工具文本
    all_tools_text = load_all_tools_text(args.tools_file)

    # 3. 提取所有有效的算法ID
    valid_algorithm_ids = extract_algorithm_ids_from_tools_text(all_tools_text)
    print(f"📋 有效算法ID数量: {len(valid_algorithm_ids)}")

    # 4. 创建LLM客户端
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
        "no_algorithm_id_found": 0,
        "exception": 0
    }
    error_cases = {
        "llm_failed": [],
        "no_algorithm_id_found": [],
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
            all_tools_text=all_tools_text,
            valid_algorithm_ids=valid_algorithm_ids,
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
                    print(f"   选中ID: {result['selected_algorithm_id']}")
                else:
                    # 大模型识别错了
                    error_cases["wrong_prediction"].append({
                        "case": case_desc,
                        "ground_truth_ids": result.get("ground_truth_ids", []),
                        "selected_algorithm_id": result["selected_algorithm_id"]
                    })
                    print(f"   ❌ 错误")
                    print(f"   正确ID: {result.get('ground_truth_ids', [])}")
                    print(f"   选中ID: {result['selected_algorithm_id']}")
            else:
                error_type = result["error_type"]
                error_stats[error_type] += 1
                error_cases[error_type].append({
                    "case": case_desc,
                    "ground_truth_ids": result.get("ground_truth_ids", []),
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

    # 记录整体结束时间（墙钟时间）
    overall_end_time = time.time()
    overall_processing_time = overall_end_time - overall_start_time

    # 6. 统计结果
    print("\n" + "=" * 80)
    print("📊 评估结果统计")
    print("=" * 80)
    print(f"总案例数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {correct_count / total_count * 100:.2f}%")

    # 处理时间统计
    print(f"\n⏱️  处理时间统计:")
    print(f"  总执行时间（墙钟时间）: {overall_processing_time:.2f}秒")

    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        sum_time = sum(processing_times)
        print(f"  平均时间: {avg_time:.2f}秒/案例")
        print(f"  最快时间: {min_time:.2f}秒")
        print(f"  最慢时间: {max_time:.2f}秒")
        print(f"  各案例时间之和: {sum_time:.2f}秒")

        # 计算并行效率
        if overall_processing_time > 0:
            parallel_efficiency = sum_time / overall_processing_time
            efficiency_percentage = (parallel_efficiency / args.max_workers) * 100
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
    time_stats = {
        "overall_processing_time_seconds": overall_processing_time,
        "overall_processing_time_description": "从脚本开始评估到结束的墙钟时间"
    }
    if processing_times:
        parallel_eff = sum(processing_times) / overall_processing_time if overall_processing_time > 0 else 0
        time_stats.update({
            "average_time_per_case": sum(processing_times) / len(processing_times),
            "min_time": min(processing_times),
            "max_time": max(processing_times),
            "sum_of_all_case_times": sum(processing_times),
            "parallel_efficiency": parallel_eff,
            "parallel_efficiency_percentage": (parallel_eff / args.max_workers * 100) if args.max_workers > 0 else 0,
            "max_workers": args.max_workers
        })

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "test_file": args.test_file,
            "tools_file": args.tools_file,
            "llm_model": args.llm_model,
            "total_tools": len(valid_algorithm_ids)
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
            "no_algorithm_id_found": {
                "description": "LLM响应中未找到Algorithm ID",
                "count": len(error_cases["no_algorithm_id_found"]),
                "cases": error_cases["no_algorithm_id_found"]
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
