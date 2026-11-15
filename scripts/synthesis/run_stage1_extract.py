#!/usr/bin/env python3
"""
第一阶段结果处理入口脚本

批量提取第一阶段生成的最终有效案例，为第二阶段做准备
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
# __file__ = .../casesynthesis_qgis/core/scripts/run_stage1_extract.py
# project_root = .../qgis_new_experiment (需要4个parent)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.processors.stage1_result_batch_processor import Stage1ResultBatchProcessor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="第一阶段结果处理：提取最终有效案例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【方式1】cd 到项目根目录运行（推荐）:
  cd /home/lhy/qgis_new_experiment

  # 使用默认路径（output/test -> output/test_save）
  python casesynthesis_qgis/core/scripts/run_stage1_extract.py

  # 指定输入输出目录
  python casesynthesis_qgis/core/scripts/run_stage1_extract.py --input-dir output/test2 --output-dir output/test_save2

  # 检查目录状态（不处理）
  python casesynthesis_qgis/core/scripts/run_stage1_extract.py --input-dir output/test --check

【方式2】使用绝对路径（可在任何目录运行）:
  python casesynthesis_qgis/core/scripts/run_stage1_extract.py --input-dir output/test --output-dir output/test_save

【高级选项】:
  # 自定义并发数
  --workers 20
        """
    )

    # 必需参数
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='output/test',
        help='输入目录（第一阶段输出目录，包含各工具子目录，默认: output/test）'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output/test_save',
        help='输出目录（保存最终案例文件，默认: output/test_save）'
    )

    # 批处理配置
    batch_group = parser.add_argument_group('批处理配置')
    batch_group.add_argument(
        '--workers', '-w',
        type=int,
        default=10,
        help='并发线程数（默认: 10）'
    )

    # 其他选项
    parser.add_argument(
        '--check',
        action='store_true',
        help='检查工具目录状态（不进行处理）'
    )

    return parser.parse_args()


def check_directory_status(input_dir: str):
    """
    检查工具目录状态

    Args:
        input_dir: 输入目录
    """
    print("🔍 检查工具目录状态")
    print(f"📁 目录: {input_dir}")

    if not os.path.exists(input_dir):
        print(f"❌ 目录不存在: {input_dir}")
        return

    all_dirs = []
    valid_dirs = []

    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)

        if os.path.isdir(item_path):
            all_dirs.append(item)
            files = os.listdir(item_path)

            # 检查工作流文件
            has_scenarios = any(f.startswith('scenarios_') and f.endswith('.json') for f in files)
            has_cases = any(f.startswith('cases_') and f.endswith('.json') for f in files)
            has_check_result = any(f.startswith('check_result_') and f.endswith('.json') for f in files)
            has_final_check = any(f.startswith('final_check_') and f.endswith('.json') for f in files)
            has_revision = any(f.startswith('revision_result_') and f.endswith('.json') for f in files)
            has_workflow_report = any(f.startswith('workflow_report_') and f.endswith('.json') for f in files)

            print(f"\n📦 {item}:")
            print(f"   场景文件: {'✅' if has_scenarios else '❌'}")
            print(f"   案例文件: {'✅' if has_cases else '❌'}")
            print(f"   检查结果: {'✅' if has_check_result else '❌'}")
            print(f"   修正结果: {'✅' if has_revision else '⚪'}")
            print(f"   最终检查: {'✅' if has_final_check else '⚪'}")
            print(f"   工作流报告: {'✅' if has_workflow_report else '❌'}")

            if has_check_result:
                valid_dirs.append(item)
                print(f"   状态: ✅ 可处理")
            else:
                print(f"   状态: ❌ 不可处理（缺少检查结果）")

    print(f"\n📊 目录统计:")
    print(f"   总目录数: {len(all_dirs)}")
    print(f"   可处理: {len(valid_dirs)}")
    print(f"   不可处理: {len(all_dirs) - len(valid_dirs)}")


def main():
    """主函数"""
    args = parse_args()

    # 如果是检查模式
    if args.check:
        check_directory_status(args.input_dir)
        return 0

    # 验证输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # 打印配置信息
    print("\n" + "="*80)
    print("🚀 第一阶段结果处理配置")
    print("="*80)
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"\n⚙️  批处理配置:")
    print(f"   并发线程数: {args.workers}")
    print("="*80 + "\n")

    # 创建批量处理器
    processor = Stage1ResultBatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers
    )

    # 运行批处理
    try:
        stats = processor.run()

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
        print(f"\n\n❌ 批量处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
