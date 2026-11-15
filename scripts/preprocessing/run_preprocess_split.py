#!/usr/bin/env python3
"""
阶段1：合并、打乱、分割训练样本

将多个工具的训练样本文件合并，打乱顺序，按4:1比例划分为训练集和验证集
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.preprocessing.config import PreprocessConfig
from core.preprocessing.splitter import DataSplitter


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="合并、打乱和划分训练样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【方式1】cd 到项目根目录运行（推荐）:
  cd /home/lhy/qgis_new_experiment
  python casesynthesis_qgis/core/scripts/run_preprocess_split.py --input-dir /path/to/input --output-dir /path/to/output

【方式2】使用绝对路径（可在任何目录运行）:
  python casesynthesis_qgis/core/scripts/run_preprocess_split.py --input-dir casesynthesis_qgis/data/retrival_G8/训练数据合成过程/所有工具带参训练数据（对final的字段处理，和字符串解析） --output-dir casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试

【自定义参数】:
  # 自定义训练集比例和随机种子
  --train-ratio 0.9 --random-seed 123
        """
    )

    # 必需参数
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='输入目录（包含多个 *_training_samples.json 文件）'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='输出目录'
    )

    # 配置参数
    config_group = parser.add_argument_group('配置参数')
    config_group.add_argument(
        '--train-ratio', '-r',
        type=float,
        default=0.8,
        help='训练集比例（默认0.8，即4:1）'
    )
    config_group.add_argument(
        '--random-seed', '-s',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 验证输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # 打印配置信息
    print("\n" + "="*80)
    print("🚀 阶段1：数据合并、打乱、分割")
    print("="*80)
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📊 训练集比例: {args.train_ratio}")
    print(f"🎲 随机种子: {args.random_seed}")
    print("="*80 + "\n")

    # 创建配置
    config = PreprocessConfig(
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )

    # 创建数据分割器
    splitter = DataSplitter(config)

    # 运行分割
    try:
        stats = splitter.run(args.input_dir, args.output_dir)

        # 打印统计结果
        print("\n" + "="*80)
        print("📊 统计结果")
        print("="*80)
        print(f"✅ 总文件数: {stats['total_files']}")
        print(f"✅ 总案例数: {stats['total_cases']}")
        print(f"✅ 训练集: {stats['train_count']} 个案例")
        print(f"✅ 验证集: {stats['val_count']} 个案例")
        print("="*80 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断！")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    exit(main())
