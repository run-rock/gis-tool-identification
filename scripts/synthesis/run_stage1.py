#!/usr/bin/env python3
"""
批量处理入口脚本

使用新的多线程 BatchProcessor 进行批量案例生成
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
# __file__ = .../casesynthesis_qgis/core/scripts/run_stage1.py
# project_root = .../qgis_new_experiment (需要4个parent)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 加载 .env 文件
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

from core.processors.stage1_batch_processor import BatchProcessor
from core.config import LLMConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量处理工具案例生成（多线程版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【方式1】cd 到项目根目录运行（推荐）:
  cd /home/lhy/qgis_new_experiment

  # 测试：只处理前5个工具
  python casesynthesis_qgis/core/scripts/run_stage1.py --tools-file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json --output-dir output/test2 --limit 5

  # 处理所有工具
  python casesynthesis_qgis/core/scripts/run_stage1.py --tools-file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json --output-dir output/stage1_full

  # 断点续传：从第100个工具开始
  python casesynthesis_qgis/core/scripts/run_stage1.py --tools-file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json --output-dir output/stage1_full --start-from 100

【方式2】使用绝对路径（可在任何目录运行）:
  # 测试：只处理前5个工具
  python casesynthesis_qgis/core/scripts/run_stage1.py --tools-file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json --output-dir output/test2 --limit 5

【高级选项】:
  # 自定义并发数和案例配置
  --workers 100 --num-scenarios 3 --cases-per-scenario 3

  # Verbose模式（在终端显示所有输出）
  --verbose
        """
    )

    # 必需参数
    parser.add_argument(
        '--tools-file',
        type=str,
        required=True,
        help='工具文件路径（JSON格式）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='输出目录路径'
    )

    # LLM 配置
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
        help='API超时时间/秒（默认: 60）'
    )

    # 批处理配置
    batch_group = parser.add_argument_group('批处理配置')
    batch_group.add_argument(
        '--workers',
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
        help='从第几个工具开始处理（用于断点续传，默认: 0）'
    )
    batch_group.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理工具数量（用于测试，默认: None表示处理所有）'
    )

    # 案例生成配置
    case_group = parser.add_argument_group('案例生成配置')
    case_group.add_argument(
        '--num-scenarios',
        type=int,
        default=5,
        help='每个工具生成的场景数（默认: 5）'
    )
    case_group.add_argument(
        '--cases-per-scenario',
        type=int,
        default=5,
        help='每个场景生成的案例数（默认: 5）'
    )
    case_group.add_argument(
        '--with-parameters',
        action='store_true',
        help='在案例生成时包含工具参数（默认: False）'
    )
    case_group.add_argument(
        '--no-revision',
        action='store_true',
        help='禁用案例修正（默认: 启用修正）'
    )

    # 其他选项
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志（在终端输出所有信息，默认: 输出到日志文件）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 验证文件路径
    if not os.path.exists(args.tools_file):
        print(f"❌ 错误: 工具文件不存在: {args.tools_file}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印配置信息
    print("\n" + "="*80)
    print("🚀 批量处理配置")
    print("="*80)
    print(f"📄 工具文件: {args.tools_file}")
    print(f"📁 输出目录: {args.output_dir}")
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
    print(f"\n📋 案例配置:")
    print(f"   场景数: {args.num_scenarios}")
    print(f"   每场景案例数: {args.cases_per_scenario}")
    print(f"   总案例数/工具: {args.num_scenarios * args.cases_per_scenario}")
    print(f"   包含参数: {'是' if args.with_parameters else '否'}")
    print(f"   启用修正: {'否' if args.no_revision else '是'}")
    print(f"\n🔧 其他选项:")
    print(f"   详细模式: {'是' if args.verbose else '否（输出到日志文件）'}")
    print("="*80 + "\n")

    # 创建 LLM 配置
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout
    )

    # 创建批处理器
    processor = BatchProcessor(
        llm_config=llm_config,
        output_dir=args.output_dir,
        max_workers=args.workers,
        retry_attempts=args.retry_attempts,
        num_scenarios=args.num_scenarios,
        cases_per_scenario=args.cases_per_scenario,
        with_parameters=args.with_parameters,
        enable_revision=not args.no_revision,
        verbose=args.verbose
    )

    # 运行批处理
    try:
        stats = processor.run(
            tools_file=args.tools_file,
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
