"""
训练工具函数
"""
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback


def last_token_pooling(last_hidden_states, attention_mask):
    """
    官方推荐的last token pooling方法

    Args:
        last_hidden_states: 模型输出的最后隐藏层状态
        attention_mask: 注意力掩码

    Returns:
        pooled_output: 池化后的输出
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class TripletDataCollator:
    """
    自定义DataCollator，处理三元组数据的动态padding
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # 提取anchor, positive, negative的input_ids和attention_mask
        anchor_input_ids = [f["anchor_input_ids"] for f in features]
        anchor_attention_mask = [f["anchor_attention_mask"] for f in features]
        positive_input_ids = [f["positive_input_ids"] for f in features]
        positive_attention_mask = [f["positive_attention_mask"] for f in features]
        negative_input_ids = [f["negative_input_ids"] for f in features]
        negative_attention_mask = [f["negative_attention_mask"] for f in features]

        # 对每组进行padding（padding到该batch内的最大长度）
        def pad_sequences(input_ids, attention_masks):
            max_len = max(len(ids) for ids in input_ids)
            padded_ids = []
            padded_masks = []
            for ids, mask in zip(input_ids, attention_masks):
                pad_len = max_len - len(ids)
                # 左padding（因为使用last token pooling）
                padded_ids.append([self.tokenizer.pad_token_id] * pad_len + ids)
                padded_masks.append([0] * pad_len + mask)
            return torch.tensor(padded_ids), torch.tensor(padded_masks)

        anchor_ids, anchor_mask = pad_sequences(anchor_input_ids, anchor_attention_mask)
        pos_ids, pos_mask = pad_sequences(positive_input_ids, positive_attention_mask)
        neg_ids, neg_mask = pad_sequences(negative_input_ids, negative_attention_mask)

        return {
            "anchor_input_ids": anchor_ids,
            "anchor_attention_mask": anchor_mask,
            "positive_input_ids": pos_ids,
            "positive_attention_mask": pos_mask,
            "negative_input_ids": neg_ids,
            "negative_attention_mask": neg_mask,
        }


class CombinedLossCallback(TrainerCallback):
    """
    自定义回调函数，用于将train和eval loss画在同一张图上
    """

    def __init__(self, logs_writer: SummaryWriter):
        self.logs_writer = logs_writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次记录日志时调用"""
        if logs is not None and state.global_step > 0:
            # 记录训练损失到combined图上
            if "loss" in logs:
                self.logs_writer.add_scalar('combined_loss/train_loss', logs["loss"], state.global_step)

            # 记录验证损失到combined图上
            if "eval_loss" in logs:
                self.logs_writer.add_scalar('combined_loss/eval_loss', logs["eval_loss"], state.global_step)
