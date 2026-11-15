"""
数据加载模块
"""
import os
import ast
import random
import pandas as pd
from typing import Dict, List, Tuple
from sentence_transformers import InputExample


def lhy_process_retrieval_ducoment(documents_df):
    """
    处理检索文档，进行去重

    Args:
        documents_df: 包含文档的DataFrame

    Returns:
        ir_corpus: Dict[doc_id, document_text]
        doc_id_map: 文档ID映射
    """
    from embedding_model_train.toolbench.utils import lhy_process_retrieval_ducoment as original_func
    return original_func(documents_df)


def build_triplet_samples(data_path: str, ir_corpus: Dict, train_or_val: str) -> Tuple[List[InputExample], Dict]:
    """
    构建三元组样本

    Args:
        data_path: 数据路径
        ir_corpus: 文档语料库
        train_or_val: "train" 或 "val"

    Returns:
        triplet_samples: List[InputExample]
        ir_train_queries: Dict[qid, query]
    """
    ir_train_queries = {}
    triplet_samples = []

    # 读取查询
    train_queries_df = pd.read_csv(os.path.join(data_path, f'{train_or_val}.query.txt'), sep='\t', names=['qid', 'query'])
    for row in train_queries_df.itertuples():
        ir_train_queries[row.qid] = row.query

    # 读取标签
    labels_df = pd.read_csv(
        os.path.join(data_path, f'qrels.{train_or_val}.tsv'),
        sep='\t',
        names=['qid', 'useless', 'positive_docids', 'negative_docids']
    )

    for row in labels_df.itertuples():
        anchor = ir_train_queries[row.qid]

        # 解析positive_docids
        positive_ids = ast.literal_eval(row.positive_docids) if isinstance(row.positive_docids, str) else []
        if not positive_ids:
            continue
        positive_id = positive_ids[0]
        positive = ir_corpus[positive_id]

        # 解析negative_docids
        random_neg_ids = ast.literal_eval(row.negative_docids) if isinstance(row.negative_docids, str) else []
        for neg_id in random_neg_ids:
            negative = ir_corpus[neg_id]
            triplet_samples.append(InputExample(texts=[anchor, positive, negative]))

    return triplet_samples, ir_train_queries


def load_training_data(config, split: str = 'train'):
    """
    加载训练数据（六类三元组样本）

    Args:
        config: TrainingConfig配置对象
        split: 'train' 或 'val'

    Returns:
        triplet_examples: 合并后的三元组样本列表
    """
    random.seed(config.random_seed)

    if split == 'train':
        paths = {
            'q_tool': config.q_tool_train_path,
            'q_toolbox': config.q_toolbox_train_path,
            'tool_q': config.tool_q_train_path,
            'toolbox_q': config.toolbox_q_train_path,
            'tool_toolbox': config.tool_toolbox_path,
            'toolbox_tool': config.toolbox_tool_path,
        }
    else:  # val
        paths = {
            'q_tool': config.q_tool_val_path,
            'q_toolbox': config.q_toolbox_val_path,
            'tool_q': config.tool_q_val_path,
            'toolbox_q': config.toolbox_q_val_path,
        }

    triplet_examples = []

    # 加载案例-工具、案例-工具箱、工具-案例、工具箱-案例
    for key in ['q_tool', 'q_toolbox', 'tool_q', 'toolbox_q']:
        if key not in paths:
            continue

        documents_df = pd.read_csv(os.path.join(paths[key], 'corpus.tsv'), sep='\t')
        ir_corpus, _ = lhy_process_retrieval_ducoment(documents_df)
        samples, _ = build_triplet_samples(paths[key], ir_corpus, "train")

        print(f"{key}: {len(samples)} samples")
        triplet_examples.extend(samples)

    # 工具-工具箱和工具箱-工具数据处理（仅训练集需要分割）
    if split == 'train' and config.include_toolbox_pairs:
        # 加载工具-工具箱
        tool_toolbox_df = pd.read_csv(os.path.join(paths['tool_toolbox'], 'corpus.tsv'), sep='\t')
        tool_toolbox_corpus, _ = lhy_process_retrieval_ducoment(tool_toolbox_df)
        tool_toolbox_samples, _ = build_triplet_samples(paths['tool_toolbox'], tool_toolbox_corpus, "train")

        # 加载工具箱-工具
        toolbox_tool_df = pd.read_csv(os.path.join(paths['toolbox_tool'], 'corpus.tsv'), sep='\t')
        toolbox_tool_corpus, _ = lhy_process_retrieval_ducoment(toolbox_tool_df)
        toolbox_tool_samples, _ = build_triplet_samples(paths['toolbox_tool'], toolbox_tool_corpus, "train")

        print(f"tool_toolbox: {len(tool_toolbox_samples)} samples")
        print(f"toolbox_tool: {len(toolbox_tool_samples)} samples")

        # 合并并分割
        t_t_samples = []
        t_t_samples.extend(tool_toolbox_samples)
        t_t_samples.extend(toolbox_tool_samples)
        random.shuffle(t_t_samples)

        split_idx = int(len(t_t_samples) * config.tool_toolbox_train_ratio)
        t_t_train_set = t_t_samples[:split_idx]

        print(f"t_t_train_set: {len(t_t_train_set)} samples")
        triplet_examples.extend(t_t_train_set)

        return triplet_examples, t_t_samples[split_idx:]  # 返回训练集和验证集（t_t部分）
    else:
        return triplet_examples, None


def load_validation_data(config, t_t_val_set=None):
    """
    加载验证数据

    Args:
        config: TrainingConfig配置对象
        t_t_val_set: 工具-工具箱的验证集（从训练数据中分割出来的）

    Returns:
        triplet_examples: 合并后的三元组样本列表
    """
    triplet_examples, _ = load_training_data(config, split='val')

    # 添加工具-工具箱的验证集
    if t_t_val_set is not None:
        print(f"t_t_val_set: {len(t_t_val_set)} samples")
        triplet_examples.extend(t_t_val_set)

    return triplet_examples
