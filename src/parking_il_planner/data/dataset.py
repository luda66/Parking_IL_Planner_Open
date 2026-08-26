"""
泊车数据集管理

提供PyTorch Dataset接口，用于模型训练
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# Action mirror mapping for horizontal flip: L+↔R+, L-↔R-
_FLIP_ACTION_MAP = np.array([0, 1, 2, 5, 6, 3, 4], dtype=np.int32)  # S0,S+,S-,L+,L-,R+,R- → S0,S+,S-,R+,R-,L+,L-


class ParkingDataset(Dataset):
    """
    泊车规划数据集

    从numpy文件加载数据，提供 (image, action, state, action_history) 对
    """

    def __init__(
        self,
        data_dir: str = "data/generated",
        split: Optional[str] = None,
        val_ratio: float = 0.15,
        transform=None,
        load_images: bool = True,
        seed: int = 42,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.load_images = load_images
        self.augment = augment

        # 加载全部数据
        all_actions = np.load(os.path.join(data_dir, "actions.npy"))
        all_states = np.load(os.path.join(data_dir, "states.npy"))
        all_obstacles = np.load(os.path.join(data_dir, "obstacles.npy"))

        # Precompute action history (previous 2 actions, one-hot 14D)
        all_action_history = self._compute_action_history(all_actions, all_states)

        n_total = len(all_actions)

        if split is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n_total)
            n_val = int(n_total * val_ratio)

            if split == 'val':
                indices = indices[:n_val]
            elif split == 'train':
                indices = indices[n_val:]
            else:
                raise ValueError(f"Unknown split: {split}")

            indices = np.sort(indices)
            self.actions = all_actions[indices]
            self.states = all_states[indices]
            self.obstacles = all_obstacles[indices]
            self.action_history = all_action_history[indices]
            self._indices = indices
        else:
            self.actions = all_actions
            self.states = all_states
            self.obstacles = all_obstacles
            self.action_history = all_action_history
            self._indices = None

        if load_images:
            self._images_mmap = np.load(os.path.join(data_dir, "images.npy"), mmap_mode='r')
            if self._indices is not None:
                self._images_subset = None
            else:
                self._images_subset = self._images_mmap  # use mmap directly for full dataset
            self.images = self._images_mmap  # __getitem__ uses _indices when needed
        else:
            self._images_mmap = None
            self._images_subset = None
            self.images = None

        print(f"数据集加载完成 ({split or 'full'}): {len(self)} 个样本" +
              (" (augment=flip)" if self.augment else ""))
        print(f"  动作分布: {np.bincount(self.actions, minlength=7)}")

    def _compute_action_history(self, actions: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Precompute 14D action history (prev 2 actions as one-hot)."""
        n = len(actions)
        history = np.zeros((n, 14), dtype=np.float32)

        # Find trajectory boundaries (where target pose changes)
        targets = states[:, 3:]
        diffs = np.abs(targets[1:] - targets[:-1]).sum(axis=1)
        boundaries = set(np.where(diffs > 0.001)[0] + 1)
        boundaries.add(0)

        for i in range(n):
            # prev action (t-1)
            if i >= 1 and i not in boundaries:
                history[i, actions[i-1]] = 1.0
            # prev-prev action (t-2)
            if i >= 2 and i not in boundaries and (i-1) not in boundaries:
                history[i, 7 + actions[i-2]] = 1.0

        return history

    def __len__(self) -> int:
        if self.augment:
            return len(self.actions) * 2
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict:
        flipped = False
        if self.augment:
            if idx >= len(self.actions):
                idx = idx - len(self.actions)
                flipped = True

        action = self.actions[idx]
        state = self.states[idx].copy()
        action_hist = self.action_history[idx].copy()

        sample = {
            'action': torch.tensor(action, dtype=torch.long),
            'state': torch.tensor(state, dtype=torch.float32),
            'obstacles': torch.tensor(self.obstacles[idx], dtype=torch.float32),
            'action_history': torch.tensor(action_hist, dtype=torch.float32),
        }

        if self.load_images and self.images is not None:
            img_idx = self._indices[idx] if self._indices is not None else idx
            img = self._images_mmap[img_idx].astype(np.float32) / 255.0
            sample['image'] = torch.tensor(img, dtype=torch.float32)

        if flipped:
            sample = self._apply_flip(sample)

        if self.transform:
            if 'image' in sample:
                sample['image'] = self.transform(sample['image'])

        return sample

    def _apply_flip(self, sample: dict) -> dict:
        """Apply horizontal flip: mirror image X-axis, swap L↔R actions."""
        # Flip image horizontally (flip last dim = width)
        if 'image' in sample:
            sample['image'] = sample['image'].flip(-1)

        # Mirror action: L+↔R+, L-↔R-
        action_val = sample['action'].item()
        sample['action'] = torch.tensor(_FLIP_ACTION_MAP[action_val], dtype=torch.long)

        # Mirror action history one-hot
        ah = sample['action_history']
        new_ah = torch.zeros_like(ah)
        for offset in [0, 7]:
            for src, dst in [(3, 5), (4, 6), (5, 3), (6, 4)]:
                new_ah[offset + dst] = ah[offset + src]
            for keep in [0, 1, 2]:
                new_ah[offset + keep] = ah[offset + keep]
        sample['action_history'] = new_ah

        # Mirror state: negate y and heading (flip around x-axis)
        state = sample['state']
        state[1] = -state[1]   # y
        state[2] = -state[2]   # heading
        state[4] = -state[4]   # target_y
        state[5] = -state[5]   # target_heading
        sample['state'] = state

        return sample

    @property
    def num_classes(self) -> int:
        return 7

    @property
    def action_distribution(self) -> np.ndarray:
        return np.bincount(self.actions, minlength=7)

    def get_class_weights(self) -> torch.Tensor:
        counts = self.action_distribution.astype(np.float64)
        counts = np.maximum(counts, 1)
        weights = len(self.actions) / (self.num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    data_dir: str = "data/generated",
    batch_size: int = 64,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, ParkingDataset]:
    dataset = ParkingDataset(data_dir, load_images=True)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\n数据划分:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")

    return train_loader, val_loader, dataset
