"""
训练入口脚本

用法:
  python training/run_training.py                  # 使用默认配置
  python training/run_training.py --epochs 100     # 自定义epochs
"""
from __future__ import annotations

import argparse

# 添加项目根目录到path
import torch
from torch.utils.data import DataLoader

from parking_il_planner.config.model import MODEL_CONFIG
from parking_il_planner.config.training import TRAINING_CONFIG
from parking_il_planner.data.dataset import ParkingDataset
from parking_il_planner.models.planner_model import build_apa_model
from parking_il_planner.training.trainer import APATrainer


def main():
    """训练入口"""
    parser = argparse.ArgumentParser(description='APA模型训练')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--data-dir', type=str, default='data/generated', help='数据目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--device', type=str, default=None, help='设备(cpu/cuda)')
    parser.add_argument('--augment', action='store_true', default=True, help='启用水平翻转增强')
    parser.add_argument('--no-augment', dest='augment', action='store_false')

    args = parser.parse_args()

    # 设备
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 更新配置
    config = TRAINING_CONFIG
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    # 创建模型
    model = build_apa_model(MODEL_CONFIG, device=device)
    print(f"模型参数: {model.get_num_params():,}")

    # 加载数据
    print("\n加载数据...")
    train_dataset = ParkingDataset(args.data_dir, split='train', load_images=True, augment=args.augment)
    val_dataset = ParkingDataset(args.data_dir, split='val', load_images=True, augment=False)

    print(f"  训练集: {len(train_dataset)} 样本 (augment={args.augment})")
    print(f"  验证集: {len(val_dataset)} 样本")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 创建训练器
    trainer = APATrainer(model, config, device=device)

    # 计算并使用类别权重 + label smoothing
    class_weights = trainer.compute_class_weights(train_loader)
    label_smoothing = getattr(MODEL_CONFIG, 'label_smoothing', 0.0)
    trainer.criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
    print(f"  Label smoothing: {label_smoothing}")

    # 开始训练
    trainer.train(train_loader, val_loader, resume_from=args.resume)

    print("\n训练完成！")
    print("最佳checkpoint保存在: checkpoints/")


if __name__ == '__main__':
    main()
