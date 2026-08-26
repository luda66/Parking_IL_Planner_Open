"""
训练器模块

负责模型的完整训练流程：
- 训练循环
- 验证循环
- 损失计算
- 指标跟踪
- 学习率调度
- Checkpoint保存/加载
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from parking_il_planner.config.training import TrainingConfig
from parking_il_planner.models.planner_model import APAPlannerImitationModel


class APATrainer:
    """
    APA模仿学习训练器
    
    功能:
    - 训练/验证循环
    - 损失和指标跟踪
    - 学习率调度
    - Early stopping
    - Checkpoint管理
    """
    
    def __init__(
        self,
        model: APAPlannerImitationModel,
        config: TrainingConfig,
        device: str = 'cpu'
    ):
        """
        初始化训练器
        
        Args:
            model: APA模型
            config: 训练配置
            device: 训练设备
        """
        self.model = model
        self.config = config
        self.device = device
        
        # 损失函数（带类别权重处理不均衡）
        self._setup_loss_function()
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # 从config获取的辅助参数
        self.log_interval = getattr(config, 'log_interval', 10)
        self.save_interval = getattr(config, 'save_every', getattr(config, 'save_interval', 5))
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 0)
        self.checkpoint_dir = getattr(config, 'checkpoint_dir', getattr(config, 'save_dir', 'checkpoints'))
        self.max_grad_norm = getattr(config, 'max_grad_norm', getattr(config, 'gradient_clip', 1.0))
    
    def _setup_loss_function(self):
        """设置损失函数，处理类别不均衡"""
        # 默认使用CrossEntropyLoss，类别权重在train()中动态计算
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_class_weights(self, dataloader: DataLoader) -> torch.Tensor:
        """
        从数据加载器计算类别权重
        
        Args:
            dataloader: 训练数据加载器
        
        Returns:
            类别权重张量
        """
        print("计算类别权重...")
        class_counts = torch.zeros(7, dtype=torch.float32)
        
        for batch in dataloader:
            actions = batch['action']
            for a in actions:
                class_counts[a] += 1
        
        # 逆频率加权（用1代替1e-6避免零计数类权重爆炸）
        total = class_counts.sum()
        class_counts_safe = class_counts.clone()
        class_counts_safe[class_counts_safe < 1] = 1.0  # 零计数类按1样本算
        weights = total / class_counts_safe
        # 只对实际有样本的类加权
        zero_mask = (class_counts < 1)
        weights[zero_mask] = 0.0
        # 归一化到均值为1
        non_zero = weights[~zero_mask]
        if non_zero.numel() > 0:
            weights[~zero_mask] = weights[~zero_mask] / non_zero.mean()
        
        print(f"  类别计数: {class_counts}")
        print(f"  类别权重: {weights}")
        
        return weights.to(self.device)
    
    def _compute_state_vector(self, states: torch.Tensor) -> torch.Tensor:
        """Compute 6D relative state vector: [dx, dy, dh, dist, cos_dh, sin_dh]"""
        dx = states[:, 3] - states[:, 0]
        dy = states[:, 4] - states[:, 1]
        dh = states[:, 5] - states[:, 2]
        dist = torch.sqrt(dx**2 + dy**2)
        cos_dh = torch.cos(dh)
        sin_dh = torch.sin(dh)
        return torch.stack([dx, dy, dh, dist, cos_dh, sin_dh], dim=-1)

    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            训练指标字典
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            images = batch['image'].to(self.device)
            actions = batch['action'].to(self.device)

            # 辅助输入
            state_vector = None
            if 'state' in batch and hasattr(self.model, 'config') and getattr(self.model.config, 'use_state_vector', False):
                state_vector = self._compute_state_vector(batch['state'].to(self.device))

            action_history = None
            if 'action_history' in batch and hasattr(self.model, 'config') and getattr(self.model.config, 'use_action_history', False):
                action_history = batch['action_history'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(images, state_vector=state_vector, action_history=action_history)
            
            # 计算损失
            loss = self.criterion(logits, actions)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == actions).sum().item()
            total += images.size(0)
            
            # 打印进度
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / total
                acc = correct / total
                print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} Acc: {acc:.4f}')
        
        # 计算平均指标
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'acc': accuracy,
            'samples': total
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 每个类别的统计
        class_correct = torch.zeros(7)
        class_total = torch.zeros(7)
        
        for batch in val_loader:
            images = batch['image'].to(self.device)
            actions = batch['action'].to(self.device)

            # 辅助输入
            state_vector = None
            if 'state' in batch and hasattr(self.model, 'config') and getattr(self.model.config, 'use_state_vector', False):
                state_vector = self._compute_state_vector(batch['state'].to(self.device))

            action_history = None
            if 'action_history' in batch and hasattr(self.model, 'config') and getattr(self.model.config, 'use_action_history', False):
                action_history = batch['action_history'].to(self.device)

            # 前向传播
            logits = self.model(images, state_vector=state_vector, action_history=action_history)
            loss = self.criterion(logits, actions)
            
            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            
            correct += (predicted == actions).sum().item()
            total += images.size(0)
            
            # 每个类别的统计
            for i in range(actions.size(0)):
                label = actions[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
        
        # 计算指标
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # 每个类别的准确率
        class_acc = torch.where(
            class_total > 0,
            class_correct / class_total,
            torch.zeros(7)
        )
        
        return {
            'loss': avg_loss,
            'acc': accuracy,
            'class_acc': class_acc.tolist(),
            'samples': total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None
    ):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            resume_from: 恢复训练的checkpoint路径
        """
        # 恢复训练
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"\n{'=' * 70}")
        print("开始训练")
        print(f"{'=' * 70}")
        print(f"  训练集: {len(train_loader.dataset)} 样本")
        print(f"  验证集: {len(val_loader.dataset)} 样本")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  学习率: {self.config.learning_rate}")
        print(f"  设备: {self.device}")
        print()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch + 1
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}] LR: {lr:.6f}")
            print("-" * 70)
            
            # 训练
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            train_time = time.time() - t0
            
            print(f"\n  训练: Loss={train_metrics['loss']:.4f} "
                  f"Acc={train_metrics['acc']:.4f} "
                  f"Time={train_time:.1f}s")
            
            # 验证
            val_metrics = self.validate(val_loader)
            print(f"  验证: Loss={val_metrics['loss']:.4f} "
                  f"Acc={val_metrics['acc']:.4f}")
            
            # 打印每个类别的准确率
            action_names = ['S0', 'S+', 'S-', 'L+', 'L-', 'R+', 'R-']
            print("  类别准确率:")
            for i, (name, acc) in enumerate(zip(action_names, val_metrics['class_acc'], strict=False)):
                print(f"    {name}: {acc:.4f}", end='')
                if i % 4 == 3:
                    print()
            print()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['acc'])
            self.training_history['val_acc'].append(val_metrics['acc'])
            self.training_history['learning_rate'].append(lr)
            
            # 保存最佳checkpoint
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self.save_checkpoint('best', extra={'val_acc': val_metrics['acc']})
                print(f"  ★ 新的最佳验证准确率: {val_metrics['acc']:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_loss', extra={'val_loss': val_metrics['loss']})
            
            # 定期保存
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
            
            # Early stopping
            if self.early_stopping_patience > 0:
                # 简单实现：如果验证损失连续N个epoch不下降，则停止
                if len(self.training_history['val_loss']) > self.early_stopping_patience:
                    recent_losses = self.training_history['val_loss'][-self.early_stopping_patience:]
                    if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                        break
        
        print(f"\n{'=' * 70}")
        print("训练完成!")
        print(f"  最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"  最佳验证损失: {self.best_val_loss:.4f}")
        print(f"{'=' * 70}")
        
        # 保存最终checkpoint
        self.save_checkpoint('final')
    
    def save_checkpoint(self, name: str, extra: Optional[Dict] = None):
        """
        保存checkpoint
        
        Args:
            name: checkpoint名称
            extra: 额外信息
        """
        checkpoint_dir = self.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': {
                'model': self.model.config.__dict__,
                'training': self.config.__dict__
            }
        }
        
        if extra:
            checkpoint.update(extra)
        
        path = os.path.join(checkpoint_dir, f'{name}.pt')
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """
        加载checkpoint
        
        Args:
            path: checkpoint路径
        """
        print(f"加载checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        print(f"  ✓ 已加载epoch {self.current_epoch}")
        print(f"  ✓ 最佳验证准确率: {self.best_val_acc:.4f}")
