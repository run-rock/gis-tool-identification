#!/usr/bin/env python3
"""
评估脚本
支持评估工具和工具箱检索任务
需要加载embedding模型对问题文本进行编码

微调模型：案例-工具
python evaluation/embedding_eval/evaluate_retrieval.py \
  -v data/retrival_G8/向量库/Qwen3-Embedding-0.6b/base/train_cases_db \
  -t data/retrival_G8/测试数据/案例检索和选择/train_case_from_tool.json \
  -o data/retrival_G8/测试结果/embedding/v1-6人工校验v2/Qwen3-Embedding-0.6b/base/train_case_retrieval \
  -p evaluation \
  --model-type base \
  --base-model-path models/Qwen3-Embedding-0.6b \
  --lora-ckpt-path models/sftmodel/qwen3-embedding-0.6b/1/2026-01-18_22-00-07/checkpoint-4480 \
  --pooling-method last_token \
  --device cuda:0 \
  -k 1 3 5 \
  -w 20
  
  
python embedding_model_train/toolbench/evaluate/G8/scripts/3_evaluate.py \
    -v casesynthesis_qgis/data/retrival_G8/向量库/合并模型1/tools_db \
    -t casesynthesis_qgis/data/retrival_G8/测试数据/工具识别/tool_multi_answer.json \
    -o casesynthesis_qgis/data/retrival_G8/测试结果/对比测试/merged \
    -p evaluation \
    --model-type base \
    --base-model-path models/Qwen3-Embedding-8b_fp16_lora_merged \
    --pooling-method last_token \
    --device cuda:0 \
    -k 1 3 5 \
    -w 20
  
# 换向量库、输出路径、lora路径
python embedding_model_train/toolbench/evaluate/G8/scripts/3_evaluate.py \
  -v casesynthesis_qgis/data/retrival_G8/向量库/加前缀_四类样本_动态超参_二次微调_案例检索/tools_db \
  -t casesynthesis_qgis/data/retrival_G8/测试数据/tool_multi_answer_with_instruction.json \
  -o casesynthesis_qgis/data/retrival_G8/测试结果/微调模型/加前缀_四类样本_动态超参_二次微调_案例检索/工具 \
  -p evaluation \
  --model-type lora \
  --base-model-path models/Qwen3-Embedding-8b \
  --lora-ckpt-path models/sftmodel/qwen3-embedding-8b/加前缀_四类样本_动态超参_二次微调_案例检索/2025-11-01_09-57-08/checkpoint-250 \
  --pooling-method last_token \
  --device cuda:0 \
  -k 1 3 5 \
  -w 20
  

微调模型：案例-工具箱
python embedding_model_train/toolbench/evaluate/G8/scripts/3_evaluate.py \
  -v casesynthesis_qgis/data/retrival_G8/向量库/千问3-8b微调1/tools_db \
  -t casesynthesis_qgis/data/retrival_G8/测试数据/工具识别/tool_multi_answer_sample100.json \
  -o casesynthesis_qgis/data/retrival_G8/测试结果-embedding-100个测试抽样效果/千问3-8b微调1/工具 \
  -p evaluation \
  --model-type lora \
  --base-model-path models/Qwen3-Embedding-8b \
  --lora-ckpt-path models/sftmodel/qwen3-embedding-8b/1/2025-10-13_21-01-27/checkpoint-3200 \
  --pooling-method last_token \
  --device cuda:0 \
  -k 1 3 5 \
  -w 20
  
  
微调模型：工具-案例
python embedding_model_train/toolbench/evaluate/G8/scripts/3_evaluate.py \
  -v casesynthesis_qgis/data/retrival_G8/向量库/Qwen3-Embedding-8b加前缀_四类样本_动态超参/cases_db \
  -t casesynthesis_qgis/data/retrival_G8/测试数据/case_from_tool_multi_with_instruction.json \
  -o casesynthesis_qgis/data/retrival_G8/测试结果/微调模型/Qwen3-Embedding-8b加前缀_四类样本_动态超参/案例 \
  -p evaluation \
  --model-type lora \
  --base-model-path models/Qwen3-Embedding-8b \
  --lora-ckpt-path models/sftmodel/qwen3-embedding-8b/加前缀_四类样本_动态超参/2025-10-29_11-03-25/checkpoint-2000 \
  --pooling-method last_token \
  --device cuda:0 \
  -k 1 3 5 \
  -w 20


基模型：案例-工具
python embedding_model_train/toolbench/evaluate/G8/scripts/3_evaluate.py \
  -v casesynthesis_qgis/data/retrival_G8/向量库/Qwen3-Embedding-8b/tools_db \
  -t casesynthesis_qgis/data/retrival_G8/测试数据/tool_multi_answer_with_instruction.json \
  -o casesynthesis_qgis/data/retrival_G8/测试结果/基模型/Qwen3-Embedding-8b加前缀/工具 \
  -p evaluation \
  --model-type base \
  --base-model-path models/Qwen3-Embedding-8b \
  --pooling-method last_token \
  --device cuda:0 \
  -k 1 3 5 \
  -w 20
  
  
基模型：案例-工具箱
python embedding_model_train/toolbench/evaluate/G8/scripts/3_evaluate.py \
  -v casesynthesis_qgis/data/retrival_G8/向量库/Qwen3-Embedding-8b/toolboxes_db \
  -t casesynthesis_qgis/data/retrival_G8/测试数据/toolbox_multi_answer_with_instruction.json \
  -o casesynthesis_qgis/data/retrival_G8/测试结果/基模型/Qwen3-Embedding-8b加前缀/工具箱 \
  -p evaluation \
  --model-type base \
  --base-model-path models/Qwen3-Embedding-8b \
  --pooling-method last_token \
  --device cuda:0 \
  -k 1 3 5 \
  -w 20
  

基模型：工具-案例
python embedding_model_train/toolbench/evaluate/G8/scripts/3_evaluate.py \
  -v casesynthesis_qgis/data/retrival_G8/向量库/Qwen3-Embedding-8b/cases_db \
  -t casesynthesis_qgis/data/retrival_G8/测试数据/case_from_tool_multi.json \
  -o casesynthesis_qgis/data/retrival_G8/测试结果/基模型/千问3-8b/案例 \
  -p evaluation \
  --model-type base \
  --base-model-path models/Qwen3-Embedding-8b \
  --pooling-method last_token \
  --device cuda:7 \
  -k 1 3 5 \
  -w 20
  


"""

import json
import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入core模块
from core.agent.vector_db import FAISSVectorDatabase
from core.common.embedding_model import create_embedding_model
from evaluation.embedding_eval.evaluator import RetrievalEvaluator


def evaluate(vector_db_path: str,
             test_cases_path: str,
             output_dir: str,
             output_prefix: str,
             model_type: str,
             k_values: list = [1, 3, 5],
             failure_threshold: float = 0.3,
             max_workers: int = 4,
             use_parallel: bool = True,
             **model_kwargs):
    """
    评估检索系统

    Args:
        vector_db_path: 向量数据库路径（不包含扩展名）
        test_cases_path: 测试案例文件路径
        output_dir: 输出目录
        output_prefix: 输出文件前缀
        model_type: 模型类型 ("base" | "lora" | "api")
        k_values: 评估的k值列表
        failure_threshold: 失败案例阈值
        max_workers: 并行评估的最大线程数
        use_parallel: 是否使用并行评估
        **model_kwargs: 模型初始化参数
    """
    print("=" * 60)
    print("🎯 检索系统评估脚本")
    print("=" * 60)
    print(f"📁 向量数据库: {vector_db_path}")
    print(f"📂 测试案例: {test_cases_path}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🤖 模型类型: {model_type}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建embedding模型
    print("=" * 60)
    print("🚀 创建Embedding模型")
    print("=" * 60)
    embedding_model = create_embedding_model(model_type, **model_kwargs)
    print()

    # 2. 加载向量数据库
    print("=" * 60)
    print("📂 加载向量数据库")
    print("=" * 60)
    vector_db = FAISSVectorDatabase(embedding_model)
    vector_db.load(vector_db_path)
    print()

    # 3. 初始化评估器
    print("=" * 60)
    print("🔧 初始化评估器")
    print("=" * 60)
    evaluator = RetrievalEvaluator(vector_db, k_values=k_values, max_workers=max_workers)
    print()

    # 4. 加载测试案例
    print("=" * 60)
    print("📂 加载测试案例")
    print("=" * 60)
    test_cases = evaluator.load_test_cases(test_cases_path)
    print()

    # 5. 执行评估
    print("=" * 60)
    print("🚀 开始评估")
    print("=" * 60)
    results = evaluator.evaluate_batch(test_cases, show_progress=True, use_parallel=use_parallel)
    print()

    # 6. 保存结果
    print("=" * 60)
    print("💾 保存评估结果")
    print("=" * 60)
    output_path = os.path.join(output_dir, output_prefix)

    # 保存JSON格式
    evaluator.save_results(results, output_path, save_format="json")

    # 保存CSV格式
    evaluator.save_results(results, output_path, save_format="csv")
    print()

    # 7. 分析失败案例
    print("=" * 60)
    print("🔍 分析失败案例")
    print("=" * 60)
    failure_cases = evaluator.analyze_failure_cases(results, threshold=failure_threshold)

    # 保存失败案例
    if failure_cases:
        failure_path = os.path.join(output_dir, f"{output_prefix}_failure_cases.json")
        with open(failure_path, 'w', encoding='utf-8') as f:
            json.dump(failure_cases, f, ensure_ascii=False, indent=2)
        print(f"   失败案例已保存到: {failure_path}")
    print()

    # 8. 分析缺失答案
    print("=" * 60)
    print("🔍 分析缺失答案")
    print("=" * 60)
    missing_report = evaluator.get_missing_answers_report(test_cases)

    # 保存缺失答案报告
    missing_path = os.path.join(output_dir, f"{output_prefix}_missing_report.json")
    with open(missing_path, 'w', encoding='utf-8') as f:
        json.dump(missing_report, f, ensure_ascii=False, indent=2)
    print(f"   缺失答案报告已保存到: {missing_path}")
    print()

    # 9. 保存完整结果
    print("=" * 60)
    print("💾 保存完整结果")
    print("=" * 60)
    complete_results = {
        "evaluation_results": results,
        "failure_cases": failure_cases,
        "missing_answers_report": missing_report
    }

    complete_path = os.path.join(output_dir, f"{output_prefix}_complete.json")
    with open(complete_path, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, ensure_ascii=False, indent=2)
    print(f"✅ 完整结果已保存到: {complete_path}")
    print()

    print("=" * 60)
    print("🎉 评估完成！")
    print("=" * 60)
    print(f"📂 结果保存在: {output_dir}")
    print(f"   {output_prefix}.json")
    print(f"   {output_prefix}_summary.csv")
    print(f"   {output_prefix}_detailed.csv")
    print(f"   {output_prefix}_failure_cases.json")
    print(f"   {output_prefix}_missing_report.json")
    print(f"   {output_prefix}_complete.json")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="检索系统评估脚本")

    # 必需参数
    parser.add_argument(
        "-v", "--vector-db",
        required=True,
        help="向量数据库路径（不包含扩展名，如 /path/to/tools_db）"
    )
    parser.add_argument(
        "-t", "--test-cases",
        required=True,
        help="测试案例文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出目录"
    )

    # 模型类型参数
    parser.add_argument(
        "--model-type",
        choices=["base", "lora", "api"],
        required=True,
        help="模型类型: base=基模型, lora=LoRA模型, api=API模型"
    )

    # 基模型和LoRA模型共用参数
    parser.add_argument(
        "--base-model-path",
        help="基模型路径（base和lora模型都需要）"
    )
    parser.add_argument(
        "--pooling-method",
        choices=["last_token", "mean"],
        default="last_token",
        help="池化方法（默认: last_token）"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="设备（默认: cuda:0）"
    )

    # LoRA模型特有参数
    parser.add_argument(
        "--lora-ckpt-path",
        help="LoRA checkpoint路径（仅lora模型需要）"
    )

    # API模型参数
    parser.add_argument(
        "--api-provider",
        choices=["openai", "zhipu", "dashscope", "custom"],
        help="API提供商（仅api模型需要）"
    )
    parser.add_argument(
        "--api-key",
        help="API密钥（仅api模型需要）"
    )
    parser.add_argument(
        "--api-model",
        help="API模型名称（仅api模型需要）"
    )
    parser.add_argument(
        "--api-base-url",
        help="API基础URL（可选）"
    )

    # 评估参数
    parser.add_argument(
        "-p", "--prefix",
        default="evaluation",
        help="输出文件前缀（默认: evaluation）"
    )
    parser.add_argument(
        "-k", "--k-values",
        type=int,
        nargs='+',
        default=[1, 3, 5],
        help="评估的k值列表（默认: 1 3 5）"
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.3,
        help="失败案例阈值（默认: 0.3）"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="并行评估的最大线程数（默认: 4）"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="禁用并行评估"
    )

    args = parser.parse_args()

    # 验证参数
    if args.model_type == "base" and not args.base_model_path:
        parser.error("--base-model-path is required for base model")

    if args.model_type == "lora":
        if not args.base_model_path:
            parser.error("--base-model-path is required for lora model")
        if not args.lora_ckpt_path:
            parser.error("--lora-ckpt-path is required for lora model")

    if args.model_type == "api":
        if not args.api_provider:
            parser.error("--api-provider is required for api model")
        if not args.api_key:
            parser.error("--api-key is required for api model")
        if not args.api_model:
            parser.error("--api-model is required for api model")

    # 构建模型参数
    model_kwargs = {}

    if args.model_type in ["base", "lora"]:
        model_kwargs["model_path" if args.model_type == "base" else "base_model_path"] = args.base_model_path
        model_kwargs["pooling_method"] = args.pooling_method
        model_kwargs["device"] = args.device

    if args.model_type == "lora":
        model_kwargs["lora_ckpt_path"] = args.lora_ckpt_path

    if args.model_type == "api":
        model_kwargs["api_provider"] = args.api_provider
        model_kwargs["api_key"] = args.api_key
        model_kwargs["api_model"] = args.api_model
        if args.api_base_url:
            model_kwargs["api_base_url"] = args.api_base_url

    try:
        evaluate(
            vector_db_path=args.vector_db,
            test_cases_path=args.test_cases,
            output_dir=args.output,
            output_prefix=args.prefix,
            model_type=args.model_type,
            k_values=args.k_values,
            failure_threshold=args.failure_threshold,
            max_workers=args.workers,
            use_parallel=not args.no_parallel,
            **model_kwargs
        )
        return 0
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
