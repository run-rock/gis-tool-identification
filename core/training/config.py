"""
训练配置
"""
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class TrainingConfig:
    """训练配置"""

    model_path: str = "Qwen/Qwen3-Embedding-8B"
    data_root: str = "data/retrival_G8/训练数据"
    include_toolbox_pairs: bool = False

    q_tool_train_path: str = field(init=False)
    q_toolbox_train_path: str = field(init=False)
    tool_q_train_path: str = field(init=False)
    toolbox_q_train_path: str = field(init=False)
    tool_toolbox_path: str = field(init=False)
    toolbox_tool_path: str = field(init=False)
    q_tool_val_path: str = field(init=False)
    q_toolbox_val_path: str = field(init=False)
    tool_q_val_path: str = field(init=False)
    toolbox_q_val_path: str = field(init=False)
    # 训练参数
    num_epochs: int = 20
    train_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: float = 0.025
    max_seq_length: int = 4096

    # LoRA参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])

    # 损失函数参数
    triplet_margin: float = 0.11
    contrastive_threshold: float = 0.61
    triplet_weight: float = 0.5
    contrastive_weight: float = 0.5

    # 评估和保存参数
    eval_steps: int = 40
    save_steps: int = 40
    save_total_limit: int = 2000
    logging_steps: int = 5

    # 工具-工具箱样本分割比例
    tool_toolbox_train_ratio: float = 0.8

    # 随机种子
    random_seed: int = 42

    def __post_init__(self):
        root = Path(self.data_root)
        self.q_tool_train_path = str(root / "案例-工具/训练")
        self.q_toolbox_train_path = str(root / "案例-工具箱/训练")
        self.tool_q_train_path = str(root / "工具-案例/训练")
        self.toolbox_q_train_path = str(root / "工具箱-案例/训练")
        self.tool_toolbox_path = str(root / "工具-工具箱")
        self.toolbox_tool_path = str(root / "工具箱-工具")
        self.q_tool_val_path = str(root / "案例-工具/验证")
        self.q_toolbox_val_path = str(root / "案例-工具箱/验证")
        self.tool_q_val_path = str(root / "工具-案例/验证")
        self.toolbox_q_val_path = str(root / "工具箱-案例/验证")
