"""
阶段3：JSONL到CSV格式转换

将三元组JSONL文件转换为训练框架需要的CSV/TSV格式。
输出三个文件：
- corpus.tsv: 文档库（docid, document_content）
- train.query.txt: 查询集合（qid, query_text）
- qrels.train.tsv: 标签关系（qid, 0, positive_docids, negative_docids）
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Literal
import pandas as pd
from tqdm import tqdm

from .formatters import format_toolbox_info


@dataclass
class ConversionConfig:
    """
    转换配置

    用于指定不同样本类型的字段映射和格式化需求
    """
    # 样本类型标识
    sample_type: Literal[
        'case_tool', 'case_toolbox',
        'tool_case', 'toolbox_case',
        'tool_toolbox', 'toolbox_tool'
    ]

    # 字段映射
    query_field: str           # query字段名：'case_description', 'tool', 'toolbox'
    positive_field: str        # 正样本字段名：'correct_tool', 'correct_toolbox', 'correct_case'
    negative_field: str        # 负样本字段名：'incorrect_tools', 'incorrect_toolboxes', 'incorrect_cases'

    # 格式化需求
    format_query: bool = False      # query是否需要格式化（toolbox作为query时需要）
    format_document: bool = False   # document是否需要格式化（toolbox作为document时需要）


# 预定义的6种转换配置
CONVERSION_CONFIGS = {
    'case_tool': ConversionConfig(
        sample_type='case_tool',
        query_field='case_description',
        positive_field='correct_tool',
        negative_field='incorrect_tools',
        format_query=False,
        format_document=False
    ),
    'case_toolbox': ConversionConfig(
        sample_type='case_toolbox',
        query_field='case_description',
        positive_field='correct_toolbox',
        negative_field='incorrect_toolboxes',
        format_query=False,
        format_document=True  # 工具箱需要格式化
    ),
    'tool_case': ConversionConfig(
        sample_type='tool_case',
        query_field='tool',
        positive_field='correct_case',
        negative_field='incorrect_cases',
        format_query=False,
        format_document=False
    ),
    'toolbox_case': ConversionConfig(
        sample_type='toolbox_case',
        query_field='toolbox',
        positive_field='correct_case',
        negative_field='incorrect_cases',
        format_query=True,  # 工具箱需要格式化
        format_document=False
    ),
    'tool_toolbox': ConversionConfig(
        sample_type='tool_toolbox',
        query_field='tool',
        positive_field='correct_toolbox',
        negative_field='incorrect_toolboxes',
        format_query=False,
        format_document=True  # 工具箱需要格式化
    ),
    'toolbox_tool': ConversionConfig(
        sample_type='toolbox_tool',
        query_field='toolbox',
        positive_field='correct_tool',
        negative_field='incorrect_tools',
        format_query=True,  # 工具箱需要格式化
        format_document=False
    ),
}


class TripletConverter:
    """
    JSONL三元组到CSV格式转换器

    完全复制原 data_preprocess_*.py 的逻辑，通过配置处理不同样本类型的差异。
    """

    def __init__(self, config: ConversionConfig):
        """
        初始化转换器

        Args:
            config: 转换配置
        """
        self.config = config

    def convert(self, jsonl_path: str, output_dir: str, save_train_json: bool = True) -> Dict[str, Any]:
        """
        转换JSONL到CSV格式

        Args:
            jsonl_path: 输入JSONL文件路径
            output_dir: 输出目录
            save_train_json: 是否保存train.json（原代码的行为）

        Returns:
            统计信息
        """
        print(f"🔄 开始转换 JSONL → CSV")
        print(f"📁 输入文件: {jsonl_path}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 样本类型: {self.config.sample_type}")
        print()

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 加载JSONL数据
        print(f"📂 加载JSONL数据...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in tqdm(f, desc="加载数据")]

        print(f"✅ 加载了 {len(data)} 条数据")

        # 保存train.json（与原代码一致）
        if save_train_json:
            train_file = output_path / "train.json"
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 保存了 train.json")

        # ID映射（与原代码完全一致）
        doc_id_map: Dict[str, int] = {}
        query_id_map: Dict[str, int] = {}

        # 文档和训练对列表
        documents: List[List] = []
        train_pairs: List = []

        # 处理数据
        print(f"\n🔄 构建ID映射和训练对...")
        for item in tqdm(data, desc="处理数据"):
            # 提取query
            query = self._extract_query(item)
            query_id = query_id_map.setdefault(query, len(query_id_map) + 1)

            # 提取正样本
            positive_doc_ids = []
            positive_doc = self._extract_positive(item)
            doc_key = json.dumps(positive_doc)
            doc_id = doc_id_map.setdefault(doc_key, len(doc_id_map) + 1)
            documents.append([doc_id, positive_doc])
            positive_doc_ids.append(doc_id)

            # 提取负样本
            negative_doc_ids = []
            negative_docs = self._extract_negatives(item)
            for neg_doc in negative_docs:
                doc_key = json.dumps(neg_doc)
                doc_id = doc_id_map.setdefault(doc_key, len(doc_id_map) + 1)
                documents.append([doc_id, neg_doc])
                negative_doc_ids.append(doc_id)

            # 创建训练对（格式与原代码完全一致）
            train_pairs.append(([query_id, query], [query_id, 0, positive_doc_ids, negative_doc_ids]))

        print(f"✅ 处理完成")
        print(f"   唯一query数: {len(query_id_map)}")
        print(f"   唯一document数: {len(doc_id_map)}")
        print(f"   训练对数: {len(train_pairs)}")

        # 转换为DataFrame（与原代码完全一致）
        train_queries, train_labels = zip(*train_pairs)

        documents_df = pd.DataFrame(documents, columns=['docid', 'document_content'])
        train_queries_df = pd.DataFrame(train_queries, columns=['qid', 'query_text'])
        train_labels_df = pd.DataFrame(train_labels, columns=['qid', 'useless', 'positive_docids', 'negative_docids'])

        # 保存CSV文件（与原代码完全一致）
        print(f"\n💾 保存CSV文件...")
        documents_df.to_csv(output_path / 'corpus.tsv', sep='\t', index=False)
        train_queries_df.to_csv(output_path / 'train.query.txt', sep='\t', index=False, header=False)
        train_labels_df.to_csv(output_path / 'qrels.train.tsv', sep='\t', index=False, header=False)

        print(f"✅ 完成!")
        print(f"   corpus.tsv: {len(documents)} 条")
        print(f"   train.query.txt: {len(train_queries)} 条")
        print(f"   qrels.train.tsv: {len(train_labels)} 条")

        return {
            'total_data': len(data),
            'unique_queries': len(query_id_map),
            'unique_documents': len(doc_id_map),
            'total_pairs': len(train_pairs),
            'output_dir': str(output_path)
        }

    def _extract_query(self, item: Dict[str, Any]) -> str:
        """提取query文本"""
        raw_query = item[self.config.query_field]

        if self.config.format_query:
            # toolbox需要格式化
            return f"Toolbox: {raw_query.get('toolbox_name', 'Unknown')}\nDescription: {raw_query.get('toolbox_description', '')}"
        else:
            return raw_query

    def _extract_positive(self, item: Dict[str, Any]) -> str:
        """提取正样本文本"""
        raw_positive = item[self.config.positive_field]

        if self.config.format_document:
            # toolbox需要格式化
            return f"Toolbox: {raw_positive.get('toolbox_name', 'Unknown')}\nDescription: {raw_positive.get('toolbox_description', '')}"
        else:
            return raw_positive

    def _extract_negatives(self, item: Dict[str, Any]) -> List[str]:
        """提取负样本文本列表"""
        raw_negatives = item[self.config.negative_field]

        if self.config.format_document:
            # toolbox需要格式化
            return [
                f"Toolbox: {neg.get('toolbox_name', 'Unknown')}\nDescription: {neg.get('toolbox_description', '')}"
                for neg in raw_negatives
            ]
        else:
            return raw_negatives
