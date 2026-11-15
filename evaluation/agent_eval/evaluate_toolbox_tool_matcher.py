#!/usr/bin/env python3
"""
工具识别评估脚本 v3

简化的单次识别逻辑：
1. 检索 k 个工具箱（仅用于验证）
2. 检索 l 个工具
3. LLM识别工具（一次性，不迭代）
4. 验证工具所属工具箱是否在检索结果中

统计指标：
1. 合理识别占比（工具所属工具箱在检索结果中）
2. 合理识别中工具正确率
3. 总体工具正确率

使用示例：
nohup python -u evaluation/agent_eval/evaluate_toolbox_tool_matcher.py \
    --test_file data/retrival_G8/测试数据/工具检索和识别/为了实现小样本多个大模型的测试/v1-6.json \
    --tools_db_path casesynthesis_qgis/data/retrival_G8/向量库/千问3-8b微调1/tools_db \
    --toolboxes_db_path casesynthesis_qgis/data/retrival_G8/向量库/千问3-8b微调1/toolboxes_db \
    --toolbox_tool_mapping_file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述-字符串版.json \
    --output_file data/retrival_G8/测试结果/工具识别/工具集-工具匹配-3-置信度/Qwen3-Embedding-8b/v1/人工校验测试集/json/工具集5-工具5.json \
    --model_type lora \
    --base_model_path models/Qwen3-Embedding-8b \
    --lora_ckpt_path models/sftmodel/qwen3-embedding-8b/1/2025-10-13_21-01-27/checkpoint-3200 \
    --pooling_method last_token \
    --device cuda:0 \
    --top_i_toolboxes 5 \
    --top_j_tools 5 \
    --max_workers 1 > data/retrival_G8/测试结果/工具识别/工具集-工具匹配-3-置信度/Qwen3-Embedding-8b/v1/人工校验测试集/log/工具集5-工具5.log 2>&1 &
"""

import json
import sys
import os
import argparse
import time
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入core模块
from core.common.embedding_model import create_embedding_model
from core.agent.vector_db import FAISSVectorDatabase
from core.agent.toolbox_tool_matcher_3 import ToolMatcher


def load_test_data(test_file: str) -> Dict[str, List[str]]:
    """加载测试数据"""
    print(f"📂 加载测试数据: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   测试案例数量: {len(data)}")
    return data


def load_toolbox_tool_mapping(mapping_file: str) -> Dict[str, List[str]]:
    """
    加载工具箱-工具映射关系
    格式: {toolbox_text: [tool_algorithm_ids]}
    """
    print(f"📂 加载工具箱-工具映射: {mapping_file}")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mapping = {}
    for toolbox_text, tools in data.items():
        tool_ids = []
        for tool_text in tools:
            algorithm_id = extract_algorithm_id_from_tool_text(tool_text)
            if algorithm_id:
                tool_ids.append(algorithm_id)
        mapping[toolbox_text] = tool_ids

    print(f"   工具箱数量: {len(mapping)}")
    total_tools = sum(len(tools) for tools in mapping.values())
    print(f"   工具总数: {total_tools}")

    return mapping


def extract_algorithm_id_from_tool_text(tool_text: str) -> str:
    """从工具文本中提取算法ID"""
    match = re.search(r'Algorithm ID:\s*([^\n]+)', tool_text)
    if match:
        return match.group(1).strip()
    return ""


def extract_algorithm_id(tool_text: str) -> str:
    """从工具文本中提取算法ID"""
    match = re.search(r'Algorithm ID:\s*([^\n]+)', tool_text)
    if match:
        return match.group(1).strip()
    raise ValueError(f"无法从工具文本中提取Algorithm ID: {tool_text[:200]}...")


def evaluate_single_case(case_desc: str,
                         ground_truth_tools: List[str],
                         tools_db: FAISSVectorDatabase,
                         toolboxes_db: FAISSVectorDatabase,
                         matcher: ToolMatcher,
                         toolbox_tool_mapping: Dict[str, List[str]],
                         top_i_toolboxes: int,
                         top_j_tools: int) -> Dict[str, Any]:
    """评估单个案例"""
    start_time = time.time()

    result = {
        "case": case_desc,
        "ground_truth": ground_truth_tools,  # 完整工具描述
        "success": False,
        "error_type": None,
        "error_message": None,
        "selected_tool_algorithm_id": None,
        "selected_tool_text": None,  # 新增：完整工具描述
        "candidate_tools_texts": [],  # 新增：候选工具完整描述列表
        "is_reasonable": False,
        "tool_correct": False,
        "processing_time": 0.0
    }

    try:
        # 1. 检索工具箱（仅用于验证）
        toolboxes_result = toolboxes_db.search(case_desc, k=top_i_toolboxes)
        candidate_toolboxes = [
            {
                'toolbox_text': item['text'],
                'score': item['score'],
                'index': item['index']
            }
            for item in toolboxes_result['results']
        ]

        # 2. 检索工具
        tools_result = tools_db.search(case_desc, k=top_j_tools)
        candidate_tools = [
            {
                'tool_text': item['text'],
                'score': item['score'],
                'index': item['index']
            }
            for item in tools_result['results']
        ]

        # 保存候选工具的完整文本和ID
        result["candidate_tools_texts"] = [t['tool_text'] for t in candidate_tools]
        result["candidate_tools"] = [extract_algorithm_id(t['tool_text']) for t in candidate_tools]

        # 提取ground truth的算法ID
        ground_truth_algorithm_ids = [extract_algorithm_id(gt) for gt in ground_truth_tools]

        # 3. 判断检索命中情况
        tool_retrieval_hit = any(gt_id in result["candidate_tools"] for gt_id in ground_truth_algorithm_ids)
        result["tool_retrieval_hit"] = tool_retrieval_hit

        toolbox_retrieval_hit = False
        for candidate_toolbox in candidate_toolboxes:
            toolbox_text = candidate_toolbox['toolbox_text']
            if toolbox_text in toolbox_tool_mapping:
                toolbox_tools = toolbox_tool_mapping[toolbox_text]
                if any(gt_id in toolbox_tools for gt_id in ground_truth_algorithm_ids):
                    toolbox_retrieval_hit = True
                    break
        result["toolbox_retrieval_hit"] = toolbox_retrieval_hit
        result["both_retrieval_hit"] = tool_retrieval_hit and toolbox_retrieval_hit

        # 4. 执行工具识别（单次，不迭代）
        match_result = matcher.identify(
            query=case_desc,
            candidate_toolboxes=candidate_toolboxes,
            candidate_tools=candidate_tools,
            extract_tool_algorithm_id_fn=extract_algorithm_id
        )

        if not match_result["success"]:
            result["error_type"] = "matcher_failed"
            result["error_message"] = match_result.get("error", "工具识别失败")
            result["success"] = True
            return result

        # 5. 记录结果
        result["selected_tool_algorithm_id"] = match_result["selected_tool_algorithm_id"]
        result["tool_belongs_to_toolbox"] = match_result["tool_belongs_to_toolbox"]
        result["is_reasonable"] = match_result["is_reasonable"]
        result["reason"] = match_result.get("reason", "unknown")  # 记录不合理的原因
        result["success"] = True

        # 查找选中工具的完整描述
        selected_tool_id = match_result["selected_tool_algorithm_id"]
        if selected_tool_id is not None:
            for tool in candidate_tools:
                if extract_algorithm_id(tool['tool_text']) == selected_tool_id:
                    result["selected_tool_text"] = tool['tool_text']
                    break

        # 6. 判断工具选择是否正确
        # 如果 selected_tool_algorithm_id 为 None（工具被判定为不可用），则判定为错误
        result["tool_correct"] = (selected_tool_id is not None and
                                  selected_tool_id in ground_truth_algorithm_ids)

    except Exception as e:
        result["error_type"] = "exception"
        result["error_message"] = str(e)
        result["success"] = True  # 设置success=True，表示流程执行完（即使有异常）

    # 记录处理时间
    end_time = time.time()
    result["processing_time"] = end_time - start_time

    return result


def main():
    parser = argparse.ArgumentParser(description="工具识别评估（简化版，无迭代）")

    # 必需参数
    parser.add_argument("--test_file", required=True, help="测试数据文件路径")
    parser.add_argument("--tools_db_path", required=True, help="工具向量库路径")
    parser.add_argument("--toolboxes_db_path", required=True, help="工具箱向量库路径")
    parser.add_argument("--toolbox_tool_mapping_file", required=True, help="工具箱-工具映射文件路径")
    parser.add_argument("--output_file", required=True, help="输出文件路径")

    # 嵌入模型参数
    parser.add_argument("--model_type", required=True, choices=["base", "lora"], help="模型类型")
    parser.add_argument("--base_model_path", required=True, help="基础模型路径")
    parser.add_argument("--lora_ckpt_path", help="LoRA checkpoint路径（model_type=lora时必需）")
    parser.add_argument("--pooling_method", default="last_token", help="池化方法")
    parser.add_argument("--device", default="cuda:0", help="设备")

    # 检索参数
    parser.add_argument("--top_i_toolboxes", type=int, default=3, help="检索工具箱数量")
    parser.add_argument("--top_j_tools", type=int, default=3, help="检索工具数量")

    # LLM参数
    parser.add_argument("--llm_provider", default="qwen3", help="LLM提供商: qwen3 | YINLI_AI | ，gei")
    parser.add_argument("--llm_model", default="qwen3-32b-gguf", help="LLM模型")

    # 并发参数
    parser.add_argument("--max_workers", type=int, default=50, help="最大并发数")

    args = parser.parse_args()

    # 加载环境变量
    load_dotenv()

    print("=" * 80)
    print("工具识别评估系统 v3（简化版）")
    print("=" * 80)

    # 1. 加载测试数据
    test_data = load_test_data(args.test_file)

    # 2. 加载工具箱-工具映射
    toolbox_tool_mapping = load_toolbox_tool_mapping(args.toolbox_tool_mapping_file)

    # 3. 创建嵌入模型
    print("\n" + "=" * 80)
    print("🔧 创建嵌入模型")
    print("=" * 80)
    # 根据模型类型准备参数
    model_kwargs = {
        "pooling_method": args.pooling_method,
        "device": args.device
    }

    if args.model_type == "base":
        model_kwargs["model_path"] = args.base_model_path
    elif args.model_type == "lora":
        model_kwargs["base_model_path"] = args.base_model_path
        model_kwargs["lora_ckpt_path"] = args.lora_ckpt_path

    embedding_model = create_embedding_model(
        model_type=args.model_type,
        **model_kwargs
    )
    print("✅ 嵌入模型创建成功")

    # 4. 加载向量库
    print("\n" + "=" * 80)
    print("📚 加载向量库")
    print("=" * 80)

    print(f"加载工具向量库: {args.tools_db_path}")
    tools_db = FAISSVectorDatabase(embedding_model)
    tools_db.load(args.tools_db_path)
    print(f"✅ 工具库加载成功，共 {len(tools_db.texts)} 个工具")

    print(f"加载工具箱向量库: {args.toolboxes_db_path}")
    toolboxes_db = FAISSVectorDatabase(embedding_model)
    toolboxes_db.load(args.toolboxes_db_path)
    print(f"✅ 工具箱库加载成功，共 {len(toolboxes_db.texts)} 个工具箱")

    # 5. 创建工具识别器
    print("\n" + "=" * 80)
    print("🤖 创建工具识别器")
    print("=" * 80)

    # 使用新的初始化方式（支持自动配置）
    matcher = ToolMatcher(
        toolbox_tool_mapping=toolbox_tool_mapping,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        use_no_think=(args.llm_provider == "qwen3"),
        temperature=0.3,
        verbose=False
    )
    print("✅ 工具识别器创建成功")

    # 6. 评估
    print("\n" + "=" * 80)
    print("🚀 开始评估")
    print("=" * 80)
    print(f"测试案例数: {len(test_data)}")
    print(f"最大并发数: {args.max_workers}")
    print(f"检索工具箱数: {args.top_i_toolboxes}")
    print(f"检索工具数: {args.top_j_tools}")

    overall_start_time = time.time()

    all_results = []
    processing_times = []
    error_stats = {
        "empty_candidates": 0,
        "matcher_failed": 0,
        "exception": 0
    }
    error_cases = {
        "empty_candidates": [],
        "matcher_failed": [],
        "exception": [],
        "wrong_prediction": []
    }

    # 创建线程锁
    print_lock = threading.Lock()

    # 并发评估
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_case = {}
        for case_id, (case_desc, ground_truth_tools) in enumerate(test_data.items(), 1):
            future = executor.submit(
                evaluate_single_case,
                case_desc,
                ground_truth_tools,
                tools_db,
                toolboxes_db,
                matcher,
                toolbox_tool_mapping,
                args.top_i_toolboxes,
                args.top_j_tools
            )
            future_to_case[future] = (case_id, case_desc, ground_truth_tools)

        # 处理结果
        total_count = len(test_data)
        completed_count = 0

        for future in as_completed(future_to_case):
            case_id, case_desc, ground_truth_tools = future_to_case[future]
            completed_count += 1

            try:
                result = future.result()
                all_results.append(result)
                processing_times.append(result["processing_time"])

                with print_lock:
                    print(f"[{completed_count}/{total_count}] 评估案例...")
                    print(f"   案例: {case_desc[:100]}...")

                    if not result["success"]:
                        error_stats[result["error_type"]] += 1
                        error_cases[result["error_type"]].append({
                            "case": case_desc,
                            "error": result["error_message"]
                        })
                        print(f"   ❌ 错误: {result['error_message']}")
                    else:
                        # 检查检索是否命中
                        if not result.get("tool_retrieval_hit", False):
                            error_cases["empty_candidates"].append({
                                "case": case_desc,
                                "ground_truth": ground_truth_tools,  # 完整工具描述
                                "retrieved_tools": result.get("candidate_tools_texts", [])  # 完整工具描述
                            })

                        print(f"   合理识别: {'✅' if result['is_reasonable'] else '❌'}")
                        print(f"   工具正确: {'✅' if result['tool_correct'] else '❌'}")

                        # 记录工具识别错误
                        if not result["tool_correct"]:
                            error_cases["wrong_prediction"].append({
                                "case": case_desc,
                                "ground_truth": ground_truth_tools,  # 完整工具描述
                                "selected_tool": result.get("selected_tool_text", None),  # 完整工具描述
                                "retrieved_tools": result.get("candidate_tools_texts", [])  # 完整工具描述
                            })

                    print(f"   ⏱️  处理时间: {result['processing_time']:.2f}秒")

            except Exception as e:
                with print_lock:
                    print(f"[{completed_count}/{total_count}] 案例评估异常: {e}")
                    error_stats["exception"] += 1
                    error_cases["exception"].append({
                        "case": case_desc,
                        "error": str(e)
                    })

    # 7. 统计结果
    print("\n" + "=" * 80)
    overall_end_time = time.time()
    overall_processing_time = overall_end_time - overall_start_time

    print("📊 评估结果统计")
    print("=" * 80)

    # 统计合理识别（is_reasonable=True）的案例
    reasonable_count = sum(
        1 for result in all_results
        if result.get("success") and result.get("is_reasonable", False)
    )
    reasonable_rate = reasonable_count / total_count * 100 if total_count > 0 else 0

    unreasonable_count = total_count - reasonable_count
    unreasonable_rate = unreasonable_count / total_count * 100 if total_count > 0 else 0

    print(f"\n基本统计:")
    print(f"  总案例数: {total_count}")
    print(f"  合理识别数: {reasonable_count} ({reasonable_rate:.2f}%)")
    print(f"  不合理识别数: {unreasonable_count} ({unreasonable_rate:.2f}%)")

    # 统计不合理识别的原因分布
    tool_invalid_count = sum(
        1 for result in all_results
        if result.get("success") and result.get("reason") == "tool_invalid"
    )
    toolbox_mismatch_count = sum(
        1 for result in all_results
        if result.get("success") and result.get("reason") == "toolbox_mismatch"
    )

    if unreasonable_count > 0:
        print(f"\n  不合理识别原因分布:")
        print(f"    - 工具不可用（返回-1）: {tool_invalid_count} ({tool_invalid_count/unreasonable_count*100:.1f}%)")
        print(f"    - 工具箱不匹配: {toolbox_mismatch_count} ({toolbox_mismatch_count/unreasonable_count*100:.1f}%)")

    # ============ 合理识别中的工具正确率 ============
    print(f"\n🎯 合理识别中的工具正确率 (is_reasonable=True 的 {reasonable_count} 个问题):")

    tool_correct_in_reasonable = sum(
        1 for result in all_results
        if result.get("success") and result.get("is_reasonable", False) and result.get("tool_correct", False)
    )
    tool_accuracy_in_reasonable = tool_correct_in_reasonable / reasonable_count * 100 if reasonable_count > 0 else 0

    print(f"  工具选择正确数: {tool_correct_in_reasonable}")
    print(f"  工具选择准确率: {tool_accuracy_in_reasonable:.2f}%")

    # ============ 总体工具正确率 ============
    print(f"\n🌐 总体工具正确率 (所有 {total_count} 个问题):")

    tool_correct_overall = sum(
        1 for result in all_results
        if result.get("success") and result.get("tool_correct", False)
    )
    tool_accuracy_overall = tool_correct_overall / total_count * 100 if total_count > 0 else 0

    print(f"  工具选择正确数: {tool_correct_overall}")
    print(f"  工具选择准确率: {tool_accuracy_overall:.2f}%")

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

    # 8. 保存结果
    print("\n" + "=" * 80)
    print(f"💾 保存结果到: {args.output_file}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 计算时间统计
    time_stats = {}
    if processing_times:
        time_stats = {
            "average_time_per_case": avg_time,
            "min_time": min_time,
            "max_time": max_time,
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
            "tools_db_path": args.tools_db_path,
            "toolboxes_db_path": args.toolboxes_db_path,
            "top_i_toolboxes": args.top_i_toolboxes,
            "top_j_tools": args.top_j_tools,
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "max_workers": args.max_workers
        },
        "summary": {
            "total_cases": total_count,
            "reasonable_count": reasonable_count,
            "reasonable_rate": reasonable_rate / 100,
            "unreasonable_count": unreasonable_count,
            "unreasonable_rate": unreasonable_rate / 100,
            "unreasonable_reason_distribution": {
                "tool_invalid": {
                    "count": tool_invalid_count,
                    "rate": tool_invalid_count / unreasonable_count if unreasonable_count > 0 else 0,
                    "description": "工具被判定为不可用（返回-1）"
                },
                "toolbox_mismatch": {
                    "count": toolbox_mismatch_count,
                    "rate": toolbox_mismatch_count / unreasonable_count if unreasonable_count > 0 else 0,
                    "description": "工具所属工具箱不在检索结果中"
                }
            },
            "reasonable_stats": {
                "tool_correct_count": tool_correct_in_reasonable,
                "tool_accuracy": tool_accuracy_in_reasonable / 100
            },
            "overall_stats": {
                "tool_correct_count": tool_correct_overall,
                "tool_accuracy": tool_accuracy_overall / 100
            },
            "error_stats": {
                **error_stats,
                "wrong_prediction": wrong_prediction_count
            },
            "processing_time_stats": time_stats
        },
        "error_cases": {
            "empty_candidates": {
                "description": "检索器没检索到正确答案",
                "count": len(error_cases["empty_candidates"]),
                "cases": error_cases["empty_candidates"]
            },
            "matcher_failed": {
                "description": "识别器调用失败",
                "count": len(error_cases["matcher_failed"]),
                "cases": error_cases["matcher_failed"]
            },
            "exception": {
                "description": "程序运行异常",
                "count": len(error_cases["exception"]),
                "cases": error_cases["exception"]
            },
            "wrong_prediction": {
                "description": "工具识别错误",
                "count": len(error_cases["wrong_prediction"]),
                "cases": error_cases["wrong_prediction"]
            }
        },
        "detailed_results": [
            {
                "case": result["case"],
                "ground_truth": result["ground_truth"],
                "retrieved_tools": result.get("candidate_tools_texts", []),
                "selected_tool": result.get("selected_tool_text", None)
            }
            for result in all_results
        ]
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("✅ 评估完成!")


if __name__ == "__main__":
    main()
