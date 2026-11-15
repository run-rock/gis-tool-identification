#!/usr/bin/env python3
"""
第二阶段批处理入口脚本

使用新的多线程 Stage2BatchProcessor 进行批量相似工具搜索和评估
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
# __file__ = .../casesynthesis_qgis/core/scripts/run_stage2.py
# project_root = .../qgis_new_experiment (需要4个parent)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 加载 .env 文件
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

from core.processors.stage2_batch_processor import Stage2BatchProcessor
from core.config import LLMConfig, EmbeddingConfig, ModelType


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="第二阶段批处理：相似工具搜索和评估（多线程版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【方式1】cd 到项目根目录运行（推荐）:
  cd /home/lhy/qgis_new_experiment

  # 测试：只处理前5个工具
  python casesynthesis_qgis/core/scripts/run_stage2.py --input-dir output/test_save2 --output-dir output/stage2_output --limit 5

  # 处理所有工具
  python casesynthesis_qgis/core/scripts/run_stage2.py --input-dir output/test_save --output-dir output/stage2_output

  # 使用已有测试数据
  python casesynthesis_qgis/core/scripts/run_stage2.py --input-dir casesynthesis_qgis/data/retrival_G8/测试数据合成过程/save_case_test测试集构建所有工具带参 --output-dir output/stage2_test --limit 5

【方式2】使用绝对路径（可在任何目录运行）:
  python casesynthesis_qgis/core/scripts/run_stage2.py --input-dir output/test_save --output-dir output/stage2_output --limit 5

【高级选项】:
  # 自定义并发数和搜索数量
  --workers 100 --top-k 5

  # 自定义向量数据库路径
  --vector-db-path /path/to/custom_vector_db

  # 详细模式
  --verbose
        """
    )

    # 必需参数
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='输入目录（包含final_cases_*.json文件，第一阶段的输出）'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='输出目录'
    )

    # 向量数据库配置
    vector_group = parser.add_argument_group('向量数据库配置')
    vector_group.add_argument(
        '--vector-db-path',
        type=str,
        default='data/retrival_G8/向量库/Qwen3-Embedding-8b/tools_db',
        help='向量数据库路径（不含扩展名，默认使用tools_db）'
    )

    # Embedding配置
    emb_group = parser.add_argument_group('Embedding配置')
    emb_group.add_argument(
        '--embedding-model',
        type=str,
        default='models/Qwen3-Embedding-8b',
        help='Embedding模型路径（默认: models/Qwen3-Embedding-8b）'
    )
    emb_group.add_argument(
        '--embedding-device',
        type=str,
        default='cuda:0',
        help='Embedding模型设备（默认: cuda:0）'
    )
    emb_group.add_argument(
        '--embedding-batch-size',
        type=int,
        default=16,
        help='Embedding批大小（默认: 16）'
    )

    # LLM配置
    llm_group = parser.add_argument_group('LLM配置')
    llm_group.add_argument(
        '--model',
        type=str,
        default='qwen3-32b-gguf',
        help='LLM 模型名称（默认: qwen3-32b-gguf）'
    )
    llm_group.add_argument(
        '--api-key',
        type=str,
        default=os.getenv("QWEN3_KEY"),
        help='API密钥（默认: 从环境变量QWEN3_KEY读取）'
    )
    llm_group.add_argument(
        '--base-url',
        type=str,
        default='http://localhost:8000/v1',
        help='API基础URL（默认: http://localhost:8000/v1）'
    )
    llm_group.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='最大token数（默认: 4096）'
    )
    llm_group.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='API超时/秒（默认: 60）'
    )

    # 批处理配置
    batch_group = parser.add_argument_group('批处理配置')
    batch_group.add_argument(
        '--workers', '-w',
        type=int,
        default=50,
        help='并发线程数（默认: 50）'
    )
    batch_group.add_argument(
        '--retry-attempts',
        type=int,
        default=2,
        help='重试次数（默认: 2）'
    )
    batch_group.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='从第几个工具开始处理（默认: 0）'
    )
    batch_group.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理工具数量（默认: None表示处理所有）'
    )

    # 搜索配置
    search_group = parser.add_argument_group('搜索配置')
    search_group.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='每个案例搜索的相似工具数量（默认: 3）'
    )

    # 其他选项
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 验证文件路径
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印配置信息
    print("\n" + "="*80)
    print("🚀 第二阶段批处理配置")
    print("="*80)
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"\n📚 向量数据库:")
    print(f"   路径: {args.vector_db_path}")
    print(f"\n🤖 Embedding配置:")
    print(f"   模型: {args.embedding_model}")
    print(f"   设备: {args.embedding_device}")
    print(f"   批大小: {args.embedding_batch_size}")
    print(f"\n🤖 LLM配置:")
    print(f"   模型: {args.model}")
    print(f"   API URL: {args.base_url}")
    print(f"   最大tokens: {args.max_tokens}")
    print(f"   超时: {args.timeout}秒")
    print(f"\n⚙️  批处理配置:")
    print(f"   并发线程数: {args.workers}")
    print(f"   重试次数: {args.retry_attempts}")
    print(f"   开始位置: {args.start_from}")
    print(f"   处理限制: {args.limit if args.limit else '无限制'}")
    print(f"\n🔍 搜索配置:")
    print(f"   每案例搜索工具数: {args.top_k}")
    print(f"\n🔧 其他选项:")
    print(f"   详细模式: {'是' if args.verbose else '否'}")
    print("="*80 + "\n")

    # 创建 Embedding 配置
    embedding_config = EmbeddingConfig(
        model_type=ModelType.BASE,
        base_model_path=args.embedding_model,
        device=args.embedding_device,
        pooling_method="last_token",
        batch_size=args.embedding_batch_size,
        max_length=8192,
        normalize=True
    )

    # 创建 LLM 配置
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout
    )

    # 创建批处理器
    processor = Stage2BatchProcessor(
        llm_config=llm_config,
        embedding_config=embedding_config,
        vector_db_path=args.vector_db_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
        retry_attempts=args.retry_attempts,
        top_k=args.top_k,
        verbose=args.verbose
    )

    # 运行批处理
    try:
        stats = processor.run(
            start_from=args.start_from,
            limit=args.limit
        )

        # 根据结果返回退出码
        if stats['failed'] > 0:
            print(f"\n⚠️  警告: 有 {stats['failed']} 个工具处理失败")
            sys.exit(1)
        else:
            print(f"\n✅ 全部完成！所有 {stats['successful']} 个工具处理成功")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断！正在保存当前进度...")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 批处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
