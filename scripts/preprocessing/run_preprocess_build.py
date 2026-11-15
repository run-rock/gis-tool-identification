#!/usr/bin/env python3
"""
阶段2：构建训练样本

支持6种样本类型：
- case_tool: 案例 → 工具
- case_toolbox: 案例 → 工具箱
- tool_case: 工具 → 案例
- toolbox_case: 工具箱 → 案例
- tool_toolbox: 工具 → 工具箱
- toolbox_tool: 工具箱 → 工具
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.preprocessing.config import PreprocessConfig
from core.preprocessing.builders import (
    CaseToolBuilder,
    CaseToolboxBuilder,
    ToolCaseBuilder,
    ToolboxCaseBuilder,
    ToolToolboxBuilder,
    ToolboxToolBuilder,
)

# 样本类型到Builder类的映射
BUILDER_MAP = {
    'case_tool': CaseToolBuilder,
    'case_toolbox': CaseToolboxBuilder,
    'tool_case': ToolCaseBuilder,
    'toolbox_case': ToolboxCaseBuilder,
    'tool_toolbox': ToolToolboxBuilder,
    'toolbox_tool': ToolboxToolBuilder,
}

# 样本类型的描述
SAMPLE_TYPE_DESCRIPTIONS = {
    'case_tool': '案例 → 工具',
    'case_toolbox': '案例 → 工具箱',
    'tool_case': '工具 → 案例（反向）',
    'toolbox_case': '工具箱 → 案例（反向）',
    'tool_toolbox': '工具 → 工具箱',
    'toolbox_tool': '工具箱 → 工具',
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="构建训练样本（三元组格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【案例相关样本-训练集】（输入: train_samples.json）:
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t case_tool -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试-训练验证数据分割/train_samples.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/案例-工具/train_triplets.jsonl --tools-file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t case_toolbox -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试-训练验证数据分割/train_samples.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/案例-工具箱/train_triplets.jsonl --tools-file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t tool_case -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试-训练验证数据分割/train_samples.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具-案例/train_triplets.jsonl
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t toolbox_case -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试-训练验证数据分割/train_samples.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具箱-案例/train_triplets.jsonl

【案例相关样本-验证集】（输入: val_samples.json）:
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t case_tool -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试-训练验证数据分割/val_samples.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/案例-工具/val_triplets.jsonl --tools-file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t case_toolbox -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试-训练验证数据分割/val_samples.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/案例-工具箱/val_triplets.jsonl --tools-file casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t tool_case -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试-训练验证数据分割/val_samples.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具-案例/val_triplets.jsonl
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t toolbox_case -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/一阶段测试-训练验证数据分割/val_samples.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具箱-案例/val_triplets.jsonl

【工具-工具箱样本】（输入: qgis_tool_with_paramaters重写工具集描述.json）:
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t tool_toolbox -i casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具-工具箱/tool_toolbox.jsonl
  python casesynthesis_qgis/core/scripts/run_preprocess_build.py -t toolbox_tool -i casesynthesis_qgis/data/qgis_tool/qgis_tool_with_paramaters重写工具集描述.json -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具箱-工具/toolbox_tool.jsonl

支持的样本类型:
  case_tool     - 案例 → 工具
  case_toolbox  - 案例 → 工具箱
  tool_case     - 工具 → 案例（反向）
  toolbox_case  - 工具箱 → 案例（反向）
  tool_toolbox  - 工具 → 工具箱
  toolbox_tool  - 工具箱 → 工具
        """
    )

    # 必需参数
    parser.add_argument(
        '--sample-type', '-t',
        type=str,
        choices=list(BUILDER_MAP.keys()),
        required=True,
        help='样本类型'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出JSONL文件路径'
    )

    # 可选参数
    optional_group = parser.add_argument_group('可选参数')
    optional_group.add_argument(
        '--tools-file',
        type=str,
        default=None,
        help='工具数据文件路径（case_tool和case_toolbox需要）'
    )

    # 配置参数
    config_group = parser.add_argument_group('配置参数')
    config_group.add_argument(
        '--random-seed', '-s',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )
    config_group.add_argument(
        '--random-negatives',
        type=int,
        default=2,
        help='每个正样本配对的随机负样本数（默认2，用于案例相关样本）'
    )
    config_group.add_argument(
        '--tool-toolbox-negatives',
        type=int,
        default=5,
        help='工具-工具箱样本的负样本数（默认5）'
    )
    config_group.add_argument(
        '--toolbox-tool-negatives',
        type=int,
        default=5,
        help='工具箱-工具样本的负样本数（默认5）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 验证输入文件
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    # 检查必需的tools_file
    if args.sample_type in ['case_tool', 'case_toolbox'] and not args.tools_file:
        print(f"❌ 错误: {args.sample_type} 类型需要指定 --tools-file 参数")
        sys.exit(1)

    if args.tools_file and not os.path.exists(args.tools_file):
        print(f"❌ 错误: 工具文件不存在: {args.tools_file}")
        sys.exit(1)

    # 打印配置信息
    print("\n" + "="*80)
    print(f"🚀 阶段2：构建训练样本 - {SAMPLE_TYPE_DESCRIPTIONS[args.sample_type]}")
    print("="*80)
    print(f"📁 输入文件: {args.input}")
    print(f"📁 输出文件: {args.output}")
    if args.tools_file:
        print(f"📦 工具文件: {args.tools_file}")
    print(f"🎲 随机种子: {args.random_seed}")
    print("="*80 + "\n")

    # 创建配置
    config = PreprocessConfig(
        random_seed=args.random_seed,
        random_negatives_per_positive=args.random_negatives,
        tool_toolbox_negatives=args.tool_toolbox_negatives,
        toolbox_tool_negatives=args.toolbox_tool_negatives,
        tools_file=args.tools_file or ""
    )

    # 获取对应的Builder类
    BuilderClass = BUILDER_MAP[args.sample_type]
    builder = BuilderClass(config)

    # 构建样本
    try:
        # 根据样本类型传递不同的参数
        if args.sample_type in ['case_tool', 'case_toolbox']:
            stats = builder.build_samples(
                input_file=args.input,
                output_file=args.output,
                tools_file=args.tools_file
            )
        else:
            stats = builder.build_samples(
                input_file=args.input,
                output_file=args.output
            )

        # 打印统计结果
        print("\n" + "="*80)
        print("📊 统计结果")
        print("="*80)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"✅ {key}: {value:.2f}")
            else:
                print(f"✅ {key}: {value}")
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
