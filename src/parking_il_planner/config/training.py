"""
训练配置
"""
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """训练超参数配置"""
    # 基本训练参数
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-6

    # 梯度裁剪
    gradient_clip: float = 1.0

    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True

    # 检查点
    save_dir: str = "checkpoints"
    save_every: int = 5

    # 日志
    log_every: int = 10

    # 随机种子
    seed: int = 42

    # 验证集比例
    val_split: float = 0.2

    # 设备
    device: str = "auto"


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_dir: str = "data/generated"

    # 场景生成
    num_train_samples: int = 10000
    num_val_samples: int = 2000

    # 场景类型比例
    scenario_ratios: dict = field(default_factory=lambda: {
        "perpendicular": 0.4,
        "parallel": 0.4,
        "angled": 0.2,
    })


@dataclass
class EvalConfig:
    """评估配置"""
    # 评估参数
    num_test_samples: int = 1000
    max_steps: int = 800

    # 成功标准
    position_threshold: float = 0.16
    angle_threshold: float = 1.6

    # 评估频率
    eval_every: int = 5


# 默认配置
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
EVAL_CONFIG = EvalConfig()
