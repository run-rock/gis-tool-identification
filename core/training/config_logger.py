"""
训练配置记录模块

功能：
- 保存训练配置为JSON格式（机器可读）
- 保存训练配置为TXT格式（人类可读）
- 记录Git版本信息
- 记录运行环境信息
"""
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import torch


def get_git_info() -> Dict[str, Any]:
    """
    获取Git版本信息

    Returns:
        包含commit hash、branch、dirty状态的字典
    """
    git_info = {
        'commit_hash': 'unknown',
        'branch': 'unknown',
        'is_dirty': False,
        'has_git': False
    }

    try:
        # 检查是否在git仓库中
        subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            check=True,
            capture_output=True,
            timeout=5
        )
        git_info['has_git'] = True

        # 获取commit hash
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()

        # 获取branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()

        # 检查是否有未提交的修改
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['is_dirty'] = bool(result.stdout.strip())

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # Git不可用或不在git仓库中
        pass

    return git_info


def get_environment_info() -> Dict[str, Any]:
    """
    获取运行环境信息

    Returns:
        包含CUDA、GPU、PyTorch版本的字典
    """
    env_info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': 'N/A',
        'gpu_count': 0,
        'gpu_names': []
    }

    if torch.cuda.is_available():
        env_info['cuda_version'] = torch.version.cuda or 'N/A'
        env_info['gpu_count'] = torch.cuda.device_count()
        env_info['gpu_names'] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]

    return env_info


def save_training_config(
    config,  # TrainingConfig实例
    args: argparse.Namespace,
    output_dir: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    保存训练配置到JSON和TXT文件

    Args:
        config: TrainingConfig实例
        args: 命令行参数
        output_dir: 输出目录
        additional_info: 额外信息（如数据统计、样本数量等）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 获取Git和环境信息
    git_info = get_git_info()
    env_info = get_environment_info()

    # 构建完整配置字典
    config_dict = {
        "training_info": {
            "timestamp": timestamp,
            "output_dir": str(output_dir)
        },
        "model_config": {
            "model_path": config.model_path,
            "max_seq_length": config.max_seq_length
        },
        "training_params": {
            "num_epochs": config.num_epochs,
            "train_batch_size": config.train_batch_size,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "eval_steps": config.eval_steps,
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,
            "logging_steps": config.logging_steps
        },
        "lora_config": {
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules
        },
        "loss_config": {
            "triplet_margin": config.triplet_margin,
            "contrastive_threshold": config.contrastive_threshold,
            "triplet_weight": config.triplet_weight,
            "contrastive_weight": config.contrastive_weight
        },
        "data_paths": {
            "q_tool_train": config.q_tool_train_path,
            "q_toolbox_train": config.q_toolbox_train_path,
            "tool_q_train": config.tool_q_train_path,
            "toolbox_q_train": config.toolbox_q_train_path,
            "q_tool_val": config.q_tool_val_path,
            "q_toolbox_val": config.q_toolbox_val_path,
            "tool_q_val": config.tool_q_val_path,
            "toolbox_q_val": config.toolbox_q_val_path,
            "tool_toolbox": config.tool_toolbox_path,
            "toolbox_tool": config.toolbox_tool_path
        },
        "git_info": git_info,
        "environment": env_info
    }

    # 添加额外信息
    if additional_info:
        config_dict["additional_info"] = additional_info

    # 保存为JSON（机器可读）
    json_path = output_dir / "training_config.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 保存为TXT（人类可读）
    txt_path = output_dir / "training_summary.txt"
    _write_human_readable_summary(txt_path, config_dict)

    print(f"✅ 训练配置已保存:")
    print(f"   - JSON格式: {json_path}")
    print(f"   - TXT格式: {txt_path}")


def _write_human_readable_summary(output_path: Path, config_dict: Dict[str, Any]) -> None:
    """
    将配置写入人类可读的TXT文件

    Args:
        output_path: 输出文件路径
        config_dict: 配置字典
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("=" * 80 + "\n")
        f.write("训练配置摘要\n")
        f.write("=" * 80 + "\n")

        # 训练信息
        training_info = config_dict.get("training_info", {})
        f.write(f"训练时间: {training_info.get('timestamp', 'N/A')}\n")
        f.write(f"输出目录: {training_info.get('output_dir', 'N/A')}\n")
        f.write("\n")

        # 基础配置
        f.write("【基础配置】\n")
        model_config = config_dict.get("model_config", {})
        f.write(f"- 基础模型: {model_config.get('model_path', 'N/A')}\n")
        f.write(f"- 最大序列长度: {model_config.get('max_seq_length', 'N/A')}\n")

        training_params = config_dict.get("training_params", {})
        f.write(f"- 训练轮数: {training_params.get('num_epochs', 'N/A')}\n")
        f.write(f"- 批次大小: {training_params.get('train_batch_size', 'N/A')}\n")
        f.write(f"- 学习率: {training_params.get('learning_rate', 'N/A')}\n")
        f.write(f"- 预热步数: {training_params.get('warmup_steps', 'N/A')}\n")
        f.write(f"- 评估步数: {training_params.get('eval_steps', 'N/A')}\n")
        f.write(f"- 保存步数: {training_params.get('save_steps', 'N/A')}\n")
        f.write(f"- 日志步数: {training_params.get('logging_steps', 'N/A')}\n")
        f.write(f"- 保存checkpoint上限: {training_params.get('save_total_limit', 'N/A')}\n")
        f.write("\n")

        # LoRA配置
        f.write("【LoRA配置】\n")
        lora_config = config_dict.get("lora_config", {})
        f.write(f"- Rank (r): {lora_config.get('r', 'N/A')}\n")
        f.write(f"- Alpha: {lora_config.get('lora_alpha', 'N/A')}\n")
        f.write(f"- Dropout: {lora_config.get('lora_dropout', 'N/A')}\n")
        target_modules = lora_config.get('target_modules', [])
        f.write(f"- 目标模块: {', '.join(target_modules) if target_modules else 'N/A'}\n")
        f.write("\n")

        # 损失函数配置
        f.write("【损失函数配置】\n")
        loss_config = config_dict.get("loss_config", {})
        f.write(f"- 三元组损失 Margin: {loss_config.get('triplet_margin', 'N/A'):.3f}\n")
        f.write(f"- 对比损失 Threshold: {loss_config.get('contrastive_threshold', 'N/A'):.3f}\n")
        f.write(f"- 三元组损失权重: {loss_config.get('triplet_weight', 'N/A')}\n")
        f.write(f"- 对比损失权重: {loss_config.get('contrastive_weight', 'N/A')}\n")
        f.write("\n")

        # 数据路径
        f.write("【数据路径】\n")
        data_paths = config_dict.get("data_paths", {})
        f.write(f"- 案例-工具训练集: {data_paths.get('q_tool_train', 'N/A')}\n")
        f.write(f"- 案例-工具箱训练集: {data_paths.get('q_toolbox_train', 'N/A')}\n")
        f.write(f"- 工具-案例训练集: {data_paths.get('tool_q_train', 'N/A')}\n")
        f.write(f"- 工具箱-案例训练集: {data_paths.get('toolbox_q_train', 'N/A')}\n")
        f.write(f"- 案例-工具验证集: {data_paths.get('q_tool_val', 'N/A')}\n")
        f.write(f"- 案例-工具箱验证集: {data_paths.get('q_toolbox_val', 'N/A')}\n")
        f.write(f"- 工具-案例验证集: {data_paths.get('tool_q_val', 'N/A')}\n")
        f.write(f"- 工具箱-案例验证集: {data_paths.get('toolbox_q_val', 'N/A')}\n")
        f.write(f"- 工具-工具箱数据: {data_paths.get('tool_toolbox', 'N/A')}\n")
        f.write(f"- 工具箱-工具数据: {data_paths.get('toolbox_tool', 'N/A')}\n")
        f.write("\n")

        # 额外信息（如样本数量）
        additional_info = config_dict.get("additional_info", {})
        if additional_info:
            f.write("【数据统计】\n")
            for key, value in additional_info.items():
                # 格式化键名（将下划线转为空格，首字母大写）
                formatted_key = key.replace('_', ' ').title()
                f.write(f"- {formatted_key}: {value}\n")
            f.write("\n")

        # Git信息
        f.write("【代码版本】\n")
        git_info = config_dict.get("git_info", {})
        if git_info.get('has_git'):
            f.write(f"- Git Commit: {git_info.get('commit_hash', 'N/A')}\n")
            f.write(f"- Git Branch: {git_info.get('branch', 'N/A')}\n")
            status = "有未提交修改" if git_info.get('is_dirty') else "干净"
            f.write(f"- 代码状态: {status}\n")
        else:
            f.write("- Git: 不可用或不在git仓库中\n")
        f.write("\n")

        # 环境信息
        f.write("【运行环境】\n")
        env_info = config_dict.get("environment", {})
        f.write(f"- PyTorch版本: {env_info.get('pytorch_version', 'N/A')}\n")

        if env_info.get('cuda_available'):
            f.write(f"- CUDA版本: {env_info.get('cuda_version', 'N/A')}\n")
            f.write(f"- GPU数量: {env_info.get('gpu_count', 0)}\n")
            gpu_names = env_info.get('gpu_names', [])
            if gpu_names:
                for i, name in enumerate(gpu_names):
                    f.write(f"  - GPU {i}: {name}\n")
        else:
            f.write("- CUDA: 不可用\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
