"""
损失函数和自定义Trainer
"""
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import Trainer
from .utils import last_token_pooling


class TripletContrastiveLossTrainer(Trainer):
    """
    结合三元组损失和对比损失的自定义Trainer

    特点：
    - L2归一化embedding
    - 三元组损失（欧氏距离）
    - 对比损失（余弦距离）
    - 可配置的margin、threshold和权重
    - 自动修正DDP导致的层名前缀问题（移除 'module.' 前缀）
    """

    def __init__(self, triplet_margin=0.18, contrastive_threshold=0.62,
                 triplet_weight=0.5, contrastive_weight=0.5, *args, **kwargs):
        """
        Args:
            triplet_margin: 三元组损失的margin
            contrastive_threshold: 对比损失的threshold
            triplet_weight: 三元组损失的权重
            contrastive_weight: 对比损失的权重
        """
        super().__init__(*args, **kwargs)
        self.triplet_margin = triplet_margin
        self.contrastive_threshold = contrastive_threshold
        self.triplet_weight = triplet_weight
        self.contrastive_weight = contrastive_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, val=False):
        # Anchor
        anchor_out = model(
            input_ids=inputs["anchor_input_ids"],
            attention_mask=inputs["anchor_attention_mask"],
        )
        anchor_emb = last_token_pooling(anchor_out.last_hidden_state, inputs["anchor_attention_mask"])

        # Positive
        pos_out = model(
            input_ids=inputs["positive_input_ids"],
            attention_mask=inputs["positive_attention_mask"],
        )
        pos_emb = last_token_pooling(pos_out.last_hidden_state, inputs["positive_attention_mask"])

        # Negative
        neg_out = model(
            input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["negative_attention_mask"],
        )
        neg_emb = last_token_pooling(neg_out.last_hidden_state, inputs["negative_attention_mask"])

        # L2归一化，将所有向量投影到单位超球面
        anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)
        neg_emb = F.normalize(neg_emb, p=2, dim=1)

        # 欧氏距离：三元组损失 Triplet Loss
        distance_pos = torch.norm(anchor_emb - pos_emb, dim=1)
        distance_neg = torch.norm(anchor_emb - neg_emb, dim=1)
        loss_triplet = F.relu(distance_pos - distance_neg + self.triplet_margin).mean()

        # 余弦距离：对比损失 Contrastive Loss
        cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
        sim_pos = cosine_sim(anchor_emb, pos_emb)
        sim_neg = cosine_sim(anchor_emb, neg_emb)

        # 转为"距离"表示
        cos_dist_pos = 1 - sim_pos
        cos_dist_neg = 1 - sim_neg

        loss_0 = 0.5 * (F.relu(self.contrastive_threshold - cos_dist_neg).pow(2)).mean()
        loss_1 = 0.5 * (cos_dist_pos.pow(2)).mean()
        loss_contrastive = loss_0 + loss_1

        # 日志输出
        if val:
            print(f"[验证] loss_triplet: {loss_triplet:.4f}, loss_contrastive: {loss_contrastive:.4f}")
        else:
            print(f"[训练] loss_triplet: {loss_triplet:.4f}, loss_contrastive: {loss_contrastive:.4f}")

        # 加权组合
        loss = self.triplet_weight * loss_triplet + self.contrastive_weight * loss_contrastive

        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # 验证阶段只计算loss，不做logits/labels预测
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, val=True)
        return (loss, None, None)

    def _save(self, output_dir=None, state_dict=None):
        """
        重写保存方法，自动修正DDP导致的层名前缀问题

        当使用accelerator.prepare()进行DDP包装时，所有参数名会被加上'module.'前缀。
        这导致保存的LoRA adapter层名变成 'module.base_model.*' 而不是 'base_model.*'。
        该方法会在保存前自动移除 'module.' 前缀，确保保存的checkpoint可以直接加载。
        """
        # 如果使用了DDP包装，需要先unwrap获取原始模型
        # 使用accelerator的unwrap_model方法可以安全地移除DDP包装
        from accelerate import Accelerator

        if hasattr(self, 'accelerator') and isinstance(self.accelerator, Accelerator):
            # 使用accelerator的unwrap_model来获取原始模型（移除DDP包装）
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            # 临时替换self.model为unwrapped版本
            original_model = self.model
            self.model = unwrapped_model
            try:
                # 调用父类的保存方法（此时模型已经unwrapped，层名正确）
                super()._save(output_dir, state_dict)
            finally:
                # 恢复原始模型引用
                self.model = original_model
        else:
            # 如果没有使用accelerator或DDP，直接调用父类方法
            super()._save(output_dir, state_dict)
