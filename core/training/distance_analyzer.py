"""
距离分析器 - 用于分析embedding模型在三元组样本上的距离分布

功能：
- 加载量化模型（4bit）
- 批量计算embeddings（支持L2归一化）
- 计算欧氏距离和余弦距离
- 统计分析和结果输出
"""
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from accelerate import Accelerator

from .utils import last_token_pooling


class DistanceAnalyzer:
    """距离分析器"""

    def __init__(
        self,
        model_path: str,
        accelerator: Accelerator,
        batch_size: int = 128,
        max_length: int = 4096,
        use_4bit: bool = True
    ):
        """
        初始化距离分析器

        Args:
            model_path: 模型路径
            accelerator: Accelerator实例（用于多卡并行）
            batch_size: 批处理大小
            max_length: 最大序列长度
            use_4bit: 是否使用4bit量化
        """
        self.model_path = model_path
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_4bit = use_4bit

        self.cosine_sim = CosineSimilarity(dim=1, eps=1e-6)

        # 加载模型和tokenizer
        self._load_model()

    def _load_model(self):
        """加载模型和tokenizer"""
        if self.accelerator.is_main_process:
            print(f"正在加载模型: {self.model_path}")
            print(f"使用4bit量化: {self.use_4bit}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = 'left'

        if self.use_4bit:
            # 配置4bit量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map={"": self.accelerator.device},
                quantization_config=bnb_config
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = self.accelerator.prepare(self.model)

        self.model.eval()

        if self.accelerator.is_main_process:
            print(f"✅ 模型加载完成")
            print(f"设备: {self.accelerator.device}")

    def get_batch_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        批量获取embeddings（L2归一化）

        Args:
            texts: 文本列表

        Returns:
            归一化的embeddings
        """
        inputs = self.tokenizer(
            texts,
            padding=True,  # 动态padding到batch内最长序列
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = last_token_pooling(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            # L2归一化
            emb = F.normalize(emb, p=2, dim=1)

        return emb

    def compute_distances(
        self,
        triplet_samples: List,
        description: str = ""
    ) -> Dict[str, List[float]]:
        """
        计算三元组样本的距离

        Args:
            triplet_samples: 三元组样本列表
            description: 描述信息（用于进度条显示）

        Returns:
            包含各种距离的字典
        """
        distances = {
            'euclidean_pos': [],
            'euclidean_neg': [],
            'cosine_pos': [],
            'cosine_neg': [],
            'triplet_margins': []
        }

        # 创建进度条
        if self.accelerator.is_main_process:
            pbar = tqdm(
                total=len(triplet_samples),
                desc=f"  🔍 {description}" if description else "  🔍 计算距离",
                unit="样本",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            pbar = None

        # 批量处理
        for i in range(0, len(triplet_samples), self.batch_size):
            batch = triplet_samples[i:i + self.batch_size]

            # 准备批量文本
            anchor_texts = [triplet.texts[0] for triplet in batch]
            positive_texts = [triplet.texts[1] for triplet in batch]
            negative_texts = [triplet.texts[2] for triplet in batch]

            # 批量获取embeddings
            anchor_embs = self.get_batch_embeddings(anchor_texts)
            pos_embs = self.get_batch_embeddings(positive_texts)
            neg_embs = self.get_batch_embeddings(negative_texts)

            # 计算欧氏距离
            euclidean_pos = torch.norm(anchor_embs - pos_embs, dim=1).cpu().numpy()
            euclidean_neg = torch.norm(anchor_embs - neg_embs, dim=1).cpu().numpy()
            distances['euclidean_pos'].extend(euclidean_pos.tolist())
            distances['euclidean_neg'].extend(euclidean_neg.tolist())

            # 计算三元组距离差值（margin）
            euclidean_margins = euclidean_neg - euclidean_pos
            distances['triplet_margins'].extend(euclidean_margins.tolist())

            # 计算余弦距离（1 - 余弦相似度）
            sim_pos = self.cosine_sim(anchor_embs, pos_embs).cpu().numpy()
            sim_neg = self.cosine_sim(anchor_embs, neg_embs).cpu().numpy()
            distances['cosine_pos'].extend((1 - sim_pos).tolist())
            distances['cosine_neg'].extend((1 - sim_neg).tolist())

            # 更新进度条
            if pbar is not None:
                pbar.update(len(batch))

        if pbar is not None:
            pbar.close()

        return distances

    @staticmethod
    def calculate_statistics(distances: List[float], name: str = "") -> Dict:
        """
        计算距离统计信息

        Args:
            distances: 距离列表
            name: 距离类型名称

        Returns:
            统计信息字典
        """
        distances = np.array(distances)
        stats = {
            'name': name,
            'count': len(distances),
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'median': float(np.median(distances)),
        }

        # 添加百分位数
        for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
            stats[f'percentile_{p}'] = float(np.percentile(distances, p))

        return stats

    @staticmethod
    def write_statistics_file(
        output_file: Path,
        model_path: str,
        sample_info: str,
        stats: Dict[str, Dict],
        category_name: Optional[str] = None
    ):
        """
        写入统计结果到文件

        Args:
            output_file: 输出文件路径
            model_path: 模型路径
            sample_info: 样本信息描述
            stats: 统计信息字典（包含euc_pos, euc_neg, cos_pos, cos_neg, margin）
            category_name: 类别名称（可选）
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            if category_name:
                f.write(f"样本类别: {category_name}\n")
            f.write(f"基模型在训练数据上的距离分布统计\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"模型路径: {model_path}\n")
            f.write(f"{sample_info}\n\n")

            # 写入各项统计
            def write_stats(s, title):
                f.write(title + "\n")
                f.write("=" * 100 + "\n")
                f.write(f"样本数量:    {s['count']}\n")
                f.write(f"均值(Mean):  {s['mean']:.6f}\n")
                f.write(f"标准差(Std): {s['std']:.6f}\n")
                f.write(f"最小值(Min): {s['min']:.6f}\n")
                f.write(f"最大值(Max): {s['max']:.6f}\n")
                f.write(f"中位数(Median): {s['median']:.6f}\n\n")
                f.write("分位数:\n")
                for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                    f.write(f"  {p}th percentile: {s[f'percentile_{p}']:.6f}\n")
                f.write("\n")

            write_stats(stats['euc_pos'], "欧氏距离(Anchor-Positive)")
            write_stats(stats['euc_neg'], "欧氏距离(Anchor-Negative)")
            write_stats(stats['cos_pos'], "余弦距离(Anchor-Positive)")
            write_stats(stats['cos_neg'], "余弦距离(Anchor-Negative)")
            write_stats(stats['margin'], "三元组欧氏距离差值 [d(a,n) - d(a,p)]")

            # 超参数建议
            margin_30 = stats['margin']['percentile_30']
            cos_pos_80 = stats['cos_pos']['percentile_80']
            cos_neg_20 = stats['cos_neg']['percentile_20']

            f.write("=" * 100 + "\n")
            f.write("💡 损失函数超参数建议\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"【三元组损失 Triplet Loss】\n")
            f.write(f"  建议 margin = {margin_30:.3f}\n")
            f.write(f"  说明: 使用30th percentile，让30%的样本参与学习\n")
            f.write(f"       （当 d(anchor,negative) - d(anchor,positive) < margin 时产生损失）\n\n")

            f.write(f"【对比损失 Contrastive Loss】\n")
            f.write(f"  建议 threshold_pos = {cos_pos_80:.3f}\n")
            f.write(f"  建议 threshold_neg = {cos_neg_20:.3f}\n")
            f.write(f"  说明: 正样本距离 > threshold_pos 产生损失（80th percentile，20%参与学习）\n")
            f.write(f"       负样本距离 < threshold_neg 产生损失（20th percentile，20%参与学习）\n\n")

        return margin_30, cos_pos_80, cos_neg_20
