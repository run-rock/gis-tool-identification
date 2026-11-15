#!/usr/bin/env python3
"""
阶段3：JSONL到CSV格式转换

将三元组JSONL文件转换为训练框架需要的CSV/TSV格式
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.preprocessing.converter import (
    TripletConverter,
    ConversionConfig,
    CONVERSION_CONFIGS,
)

# 样本类型的描述
SAMPLE_TYPE_DESCRIPTIONS = {
    'case_tool': '案例 → 工具',
    'case_toolbox': '案例 → 工具箱',
    'tool_case': '工具 → 案例',
    'toolbox_case': '工具箱 → 案例',
    'tool_toolbox': '工具 → 工具箱',
    'toolbox_tool': '工具箱 → 工具',
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="JSONL到CSV格式转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

【案例-工具】:
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t case_tool -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/案例-工具/train_triplets.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/案例-工具/训练
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t case_tool -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/案例-工具/val_triplets.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/案例-工具/验证

【案例-工具箱】:
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t case_toolbox -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/案例-工具箱/train_triplets.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/案例-工具箱/训练
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t case_toolbox -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/案例-工具箱/val_triplets.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/案例-工具箱/验证

【工具-案例】:
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t tool_case -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具-案例/train_triplets.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/工具-案例/训练
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t tool_case -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具-案例/val_triplets.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/工具-案例/验证

【工具箱-案例】:
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t toolbox_case -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具箱-案例/train_triplets.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/工具箱-案例/训练
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t toolbox_case -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具箱-案例/val_triplets.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/工具箱-案例/验证

【工具-工具箱】:
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t tool_toolbox -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具-工具箱/tool_toolbox.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/工具-工具箱

【工具箱-工具】:
  python casesynthesis_qgis/core/scripts/run_preprocess_convert.py -t toolbox_tool -i casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/二阶段测试-三元组样本构建/工具箱-工具/toolbox_tool.jsonl -o casesynthesis_qgis/data/retrival_G8/数据预处理重构测试/三阶段测试-CSV转换/工具箱-工具

支持的样本类型:
  case_tool     - 案例 → 工具
  case_toolbox  - 案例 → 工具箱
  tool_case     - 工具 → 案例
  toolbox_case  - 工具箱 → 案例
  tool_toolbox  - 工具 → 工具箱
  toolbox_tool  - 工具箱 → 工具
        """
    )

    # 必需参数
    parser.add_argument(
        '--sample-type', '-t',
        type=str,
        choices=list(CONVERSION_CONFIGS.keys()),
        required=True,
        help='样本类型'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录'
    )

    # 可选参数
    parser.add_argument(
        '--no-train-json',
        action='store_true',
        help='不保存train.json文件'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 验证输入文件
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    # 打印配置信息
    print("\n" + "="*80)
    print(f"🚀 阶段3：JSONL → CSV 转换 - {SAMPLE_TYPE_DESCRIPTIONS[args.sample_type]}")
    print("="*80)
    print(f"📁 输入文件: {args.input}")
    print(f"📁 输出目录: {args.output}")
    print("="*80 + "\n")

    # 获取对应的配置
    config = CONVERSION_CONFIGS[args.sample_type]
    converter = TripletConverter(config)

    # 转换
    try:
        stats = converter.convert(
            jsonl_path=args.input,
            output_dir=args.output,
            save_train_json=not args.no_train_json
        )

        # 打印统计结果
        print("\n" + "="*80)
        print("📊 统计结果")
        print("="*80)
        for key, value in stats.items():
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
