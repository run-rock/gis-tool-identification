#!/usr/bin/env python3
"""
一键构建所有向量数据库（工具、工具箱、案例）
支持基模型、LoRA模型、API模型



用法示例:
# 基模型一键构建所有向量库
python evaluation/embedding_eval/build_all_vectordbs.py \
  --tool-data data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json \
  --case-data data/retrival_G8/测试数据/case_from_tool_multi.json \
  -o data/retrival_G8/向量库/Qwen3-Embedding-0.6b \
  --model-type base \
  --base-model-path models/Qwen3-Embedding-0.6b \
  --pooling-method last_token \
  --device cuda:0

# LoRA模型一键构建所有向量库
python evaluation/embedding_eval/build_all_vectordbs.py \
  --tool-data data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json \
  --case-data data/retrival_G8/测试数据/case_from_tool_multi.json \
  -o casesynthesis_qgis/data/retrival_G8/向量库/千问3-8b微调1 \
  --model-type lora \
  --base-model-path models/Qwen3-Embedding-8b \
  --lora-ckpt-path models/sftmodel/qwen3-embedding-8b/1/2025-10-13_21-01-27/checkpoint-3200 \
  --pooling-method last_token \
  --device cuda:2
  
  
python evaluation/embedding_eval/build_all_vectordbs.py \
  --tool-data data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json \
  --case-data data/retrival_G8/测试数据/工具调用/训练数据构建案例选择和参数调用/train_case_from_tool_人工校验.json \
  -o data/retrival_G8/向量库/Qwen3-Embedding-0.6b \
  --model-type lora \
  --base-model-path models/Qwen3-Embedding-0.6b \
  --lora-ckpt-path models/sftmodel/qwen3-embedding-0.6b/1/2026-01-18_22-00-07/checkpoint-4480 \
  --pooling-method last_token \
  --device cuda:3
  
  
python evaluation/embedding_eval/build_all_vectordbs.py \
  --tool-data data/qgis_tool/qgis_tool_with_paramaters重写工具集描述_人工校验.json \
  --case-data casesynthesis_qgis/data/retrival_G8/训练数据/train_case_from_tool.json \
  -o data/retrival_G8/向量库/Qwen3-Embedding-8b \
  --model-type lora \
  --base-model-path models/Qwen3-Embedding-8b \
  --lora-ckpt-path models/sftmodel/qwen3-embedding-8b/1/2025-10-13_21-01-27/checkpoint-3200 \
  --pooling-method last_token \
  --device cuda:2


# API模型一键构建所有向量库
python evaluation/embedding_eval/build_all_vectordbs.py \
  --tool-data /path/to/qgis_tool_with_paramaters.json \
  --case-data /path/to/case_from_tool_multi.json \
  -o /path/to/output_dir \
  --model-type api \
  --api-provider openai \
  --api-key YOUR_API_KEY \
  --api-model text-embedding-3-large
"""

import json
import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入core模块
from core.common.embedding_model import create_embedding_model
from core.agent.vector_db import FAISSVectorDatabase
from core.agent.data_parser import (
    extract_tools_and_toolboxes,
    extract_cases_from_test_file
)


def build_all_vectordbs(
    tool_data_file: str,
    case_data_file: str,
    output_dir: str,
    model_type: str,
    **model_kwargs
):
    """
    一键构建所有向量数据库

    Args:
        tool_data_file: 工具数据文件路径
        case_data_file: 案例数据文件路径（测试集）
        output_dir: 输出目录
        model_type: 模型类型 ("base" | "lora" | "api")
        **model_kwargs: 模型初始化参数
    """
    print("=" * 80)
    print("🎯 一键构建所有向量数据库")
    print("=" * 80)
    print(f"📁 工具数据文件: {tool_data_file}")
    print(f"📁 案例数据文件: {case_data_file}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🤖 模型类型: {model_type}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # ========== 步骤1: 创建embedding模型 ==========
    print("=" * 80)
    print("🚀 【步骤1/4】创建Embedding模型")
    print("=" * 80)
    embedding_model = create_embedding_model(model_type, **model_kwargs)
    print("✅ 模型创建成功")
    print()

    # ========== 步骤2: 构建工具和工具箱向量库 ==========
    print("=" * 80)
    print("🔧 【步骤2/4】构建工具和工具箱向量库")
    print("=" * 80)

    # 2.1 加载工具数据
    print(f"📄 读取工具数据: {tool_data_file}")
    with open(tool_data_file, 'r', encoding='utf-8') as f:
        tool_data = json.load(f)
    print(f"✅ 成功读取 {len(tool_data)} 个工具箱")

    # 2.2 提取工具和工具箱数据
    print("🔍 提取工具和工具箱数据...")
    extracted_tool_data = extract_tools_and_toolboxes(tool_data)
    print(f"✅ 提取了 {len(extracted_tool_data['tools'])} 个工具")
    print(f"✅ 提取了 {len(extracted_tool_data['toolboxes'])} 个工具箱")

    # 2.3 创建工具向量库
    print("🔧 创建工具向量库...")
    tools_db = FAISSVectorDatabase(
        embedding_model=embedding_model,
        index_type="IndexFlatIP",
        normalize_embeddings=True
    )
    tools_db.add_texts(
        extracted_tool_data["tools"],
        extracted_tool_data["tools_metadata"]
    )
    tools_db_path = os.path.join(output_dir, "tools_db")
    tools_db.save(tools_db_path)
    print(f"✅ 工具向量库创建完成: {tools_db_path}")

    # 2.4 创建工具箱向量库
    print("📦 创建工具箱向量库...")
    toolboxes_db = FAISSVectorDatabase(
        embedding_model=embedding_model,
        index_type="IndexFlatIP",
        normalize_embeddings=True
    )
    toolboxes_db.add_texts(
        extracted_tool_data["toolboxes"],
        extracted_tool_data["toolboxes_metadata"]
    )
    toolboxes_db_path = os.path.join(output_dir, "toolboxes_db")
    toolboxes_db.save(toolboxes_db_path)
    print(f"✅ 工具箱向量库创建完成: {toolboxes_db_path}")
    print()

    # ========== 步骤3: 构建案例向量库 ==========
    print("=" * 80)
    print("📚 【步骤3/4】构建案例向量库")
    print("=" * 80)

    # 3.1 提取案例数据
    print(f"📄 读取案例数据: {case_data_file}")
    extracted_case_data = extract_cases_from_test_file(case_data_file)
    print(f"✅ 提取了 {len(extracted_case_data['cases'])} 个唯一案例")

    # 3.2 创建案例向量库
    print("🔧 创建案例向量库...")
    cases_db = FAISSVectorDatabase(
        embedding_model=embedding_model,
        index_type="IndexFlatIP",
        normalize_embeddings=True
    )
    cases_db.add_texts(
        extracted_case_data["cases"],
        extracted_case_data["cases_metadata"]
    )
    cases_db_path = os.path.join(output_dir, "cases_db")
    cases_db.save(cases_db_path)
    print(f"✅ 案例向量库创建完成: {cases_db_path}")
    print()

    # ========== 步骤4: 输出统计信息 ==========
    print("=" * 80)
    print("📊 【步骤4/4】统计信息")
    print("=" * 80)
    print(f"工具向量库:   {tools_db.get_stats()}")
    print(f"工具箱向量库: {toolboxes_db.get_stats()}")
    print(f"案例向量库:   {cases_db.get_stats()}")
    print()

    print("=" * 80)
    print("🎉 所有向量数据库构建完成！")
    print("=" * 80)
    print(f"📂 保存位置: {output_dir}")
    print(f"   ├── tools_db.faiss")
    print(f"   ├── tools_db.pkl")
    print(f"   ├── toolboxes_db.faiss")
    print(f"   ├── toolboxes_db.pkl")
    print(f"   ├── cases_db.faiss")
    print(f"   └── cases_db.pkl")
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="一键构建所有向量数据库（工具、工具箱、案例）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ========== 必需参数 ==========
    parser.add_argument(
        "--tool-data",
        required=True,
        help="工具数据文件路径（qgis_tool_with_paramaters重写工具集描述.json）"
    )
    parser.add_argument(
        "--case-data",
        required=True,
        help="案例数据文件路径（case_from_tool_*.json 测试集）"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="向量库输出目录"
    )

    # ========== 模型类型参数 ==========
    parser.add_argument(
        "--model-type",
        choices=["base", "lora", "api"],
        required=True,
        help="模型类型: base=基模型, lora=LoRA模型, api=API模型"
    )

    # ========== 基模型和LoRA模型共用参数 ==========
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

    # ========== LoRA模型特有参数 ==========
    parser.add_argument(
        "--lora-ckpt-path",
        help="LoRA checkpoint路径（仅lora模型需要）"
    )

    # ========== API模型参数 ==========
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
        help="API模型名称（仅api模型需要，如: text-embedding-3-large）"
    )
    parser.add_argument(
        "--api-base-url",
        help="API基础URL（可选，用于兼容OpenAI接口的服务）"
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=100,
        help="API批次大小（默认: 100）"
    )

    args = parser.parse_args()

    # ========== 验证参数 ==========
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

    # ========== 构建模型参数 ==========
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
        if args.api_batch_size:
            model_kwargs["api_batch_size"] = args.api_batch_size

    # ========== 执行构建 ==========
    try:
        build_all_vectordbs(
            tool_data_file=args.tool_data,
            case_data_file=args.case_data,
            output_dir=args.output,
            model_type=args.model_type,
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
