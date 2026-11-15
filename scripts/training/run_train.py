"""
Embedding模型训练脚本

使用QLoRA + 三元组对比损失训练Qwen3-Embedding模型

运行示例:
    python scripts/training/run_train.py \
        --data_path /path/to/data \
        --output_path /path/to/output \
        --num_epochs 5 \
        --train_batch_size 32 \
        --learning_rate 2e-5 \
        --warmup_steps 500 \
        --max_seq_length 4096
    
        
    
    
"""
import os
import sys
import logging
import argparse
import random
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from torch.utils.tensorboard import SummaryWriter
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from sentence_transformers import LoggingHandler

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.training import (
    TrainingConfig,
    load_training_data,
    load_validation_data,
    TripletContrastiveLossTrainer,
    TripletDataCollator,
    CombinedLossCallback,
    save_training_config
)

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练Embedding模型")
    parser.add_argument("--model_path", default="Qwen/Qwen3-Embedding-8B", help="Base model ID or local directory")
    parser.add_argument("--include_toolbox_pairs", action="store_true", help="Include supplemental tool-toolbox pairs")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据目录路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="模型输出路径")
    parser.add_argument("--num_epochs", type=int, required=True,
                        help="训练轮数")
    parser.add_argument("--train_batch_size", type=int, required=True,
                        help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, required=True,
                        help="学习率")
    parser.add_argument("--warmup_steps", type=float, required=True,
                        help="预热步数")
    parser.add_argument("--max_seq_length", type=int, required=True,
                        help="最大序列长度")
    return parser.parse_args()


def tokenize_triplets(ex, tokenizer, max_length=4096):
    """
    将三元组样本tokenize

    使用动态padding：在tokenization时不padding，由DataLoader在batch时自动padding
    """
    a = tokenizer(ex["text1"], truncation=True, max_length=max_length)
    p = tokenizer(ex["text2"], truncation=True, max_length=max_length)
    n = tokenizer(ex["text3"], truncation=True, max_length=max_length)
    return {
        "anchor_input_ids": a["input_ids"],
        "anchor_attention_mask": a["attention_mask"],
        "positive_input_ids": p["input_ids"],
        "positive_attention_mask": p["attention_mask"],
        "negative_input_ids": n["input_ids"],
        "negative_attention_mask": n["attention_mask"],
    }


def examples_to_dicts(examples):
    """将InputExample转换为字典列表"""
    return [
        {"text1": ex.texts[0], "text2": ex.texts[1], "text3": ex.texts[2]}
        for ex in examples
    ]


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    # 创建配置
    config = TrainingConfig(data_root=args.data_path, model_path=args.model_path,
                            include_toolbox_pairs=args.include_toolbox_pairs)
    config.num_epochs = args.num_epochs
    config.train_batch_size = args.train_batch_size
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    config.max_seq_length = args.max_seq_length

    # 创建输出目录
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    model_save_path = os.path.join(output_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(model_save_path, exist_ok=True)

    # 创建tensorboard writer
    tensorboard_name = 'accuracy_v1'
    logs_writer = SummaryWriter(os.path.join(model_save_path, 'accuracy', 'tensorboard', tensorboard_name))

    # 加载tokenizer和模型
    logger.info("加载模型和tokenizer...")
    model_path = config.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    logger.info("已设置tokenizer使用左对齐padding用于last_token池化")

    # 配置accelerator
    ddp_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs]
    )

    # 配置4bit量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 加载量化模型
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": torch.cuda.current_device()},
        quantization_config=bnb_config
    )

    model = prepare_model_for_kbit_training(model)

    # 配置LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=config.lora_target_modules
    )
    model = get_peft_model(model, lora_config)
    model = accelerator.prepare(model)

    # 加载训练数据
    logger.info("加载训练数据...")
    train_triplet_examples, t_t_val_set = load_training_data(config, split='train')
    logger.info(f"训练样本总数: {len(train_triplet_examples)}")

    # 加载验证数据
    logger.info("加载验证数据...")
    val_triplet_examples = load_validation_data(config, t_t_val_set=t_t_val_set)
    logger.info(f"验证样本总数: {len(val_triplet_examples)}")

    # 转换为Dataset格式
    train_dicts = examples_to_dicts(train_triplet_examples)
    val_dicts = examples_to_dicts(val_triplet_examples)

    train_ds = Dataset.from_list(train_dicts)
    train_ds = train_ds.map(
        lambda ex: tokenize_triplets(ex, tokenizer, config.max_seq_length),
        remove_columns=["text1", "text2", "text3"]
    )
    logger.info(f"训练数据集大小: {len(train_ds)}")

    val_ds = Dataset.from_list(val_dicts)
    val_ds = val_ds.map(
        lambda ex: tokenize_triplets(ex, tokenizer, config.max_seq_length),
        remove_columns=["text1", "text2", "text3"]
    )
    logger.info(f"验证数据集大小: {len(val_ds)}")

    # 保存训练配置（在训练开始前）
    logger.info("\n保存训练配置...")
    save_training_config(
        config=config,
        args=args,
        output_dir=model_save_path,
        additional_info={
            'train_samples': len(train_triplet_examples),
            'val_samples': len(val_triplet_examples),
            'train_dataset_size': len(train_ds),
            'val_dataset_size': len(val_ds)
        }
    )

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_steps,
        fp16=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_dir=os.path.join(model_save_path, 'loss', 'tensorboard', 'loss_v1'),
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    # 创建自定义组件
    combined_loss_callback = CombinedLossCallback(logs_writer)
    triplet_data_collator = TripletDataCollator(tokenizer)

    # 创建trainer
    trainer = TripletContrastiveLossTrainer(
        triplet_margin=config.triplet_margin,
        contrastive_threshold=config.contrastive_threshold,
        triplet_weight=config.triplet_weight,
        contrastive_weight=config.contrastive_weight,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=triplet_data_collator,
        callbacks=[combined_loss_callback],
    )
    # 将accelerator传递给trainer，用于在保存时自动修正层名
    trainer.accelerator = accelerator

    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
