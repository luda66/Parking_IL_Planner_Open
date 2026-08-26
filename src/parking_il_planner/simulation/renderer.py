"""
图像渲染器

将场景渲染为384×384三通道图像，用于模型输入

通道定义:
  通道0 (R): 当前车辆位置
  通道1 (G): 目标位置
  通道2 (B): 障碍物位置

参数:
  分辨率: 20像素/米
  图像尺寸: 384×384
  视野: 约19.2米×19.2米
"""
from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np

from parking_il_planner.config.vehicle import VEHICLE_CONFIG
from parking_il_planner.geometry.collision import Rectangle
from parking_il_planner.geometry.kinematics import VehicleState


class SceneRenderer:
    """
    场景渲染器
    
    将车辆状态、目标位姿、障碍物渲染为3通道图像
    """

    # 渲染参数
    PIXELS_PER_METER = 20.0    # 20像素/米
    IMAGE_SIZE = 384           # 384×384
    
    # 颜色定义 (0-255)
    VEHICLE_COLOR = 255        # 红色通道 - 当前车辆
    TARGET_COLOR = 255         # 绿色通道 - 目标位置
    OBSTACLE_COLOR = 255       # 蓝色通道 - 障碍物
    
    def __init__(self, view_size: float = 16.0):
        """
        初始化
        
        Args:
            view_size: 视野大小 (米)，以车辆当前位置为中心
        """
        self.view_size = view_size
        self.half_view = view_size / 2.0
        
        # 计算偏移量，使视野中心对准图像中心
        self.offset_x = self.IMAGE_SIZE / 2
        self.offset_y = self.IMAGE_SIZE / 2

    def _world_to_pixel(self, world_x: float, world_y: float, 
                        center_x: float, center_y: float) -> Tuple[int, int]:
        """
        世界坐标 → 像素坐标
        
        以车辆当前位置为中心
        
        Args:
            world_x, world_y: 世界坐标
            center_x, center_y: 视野中心（车辆位置）
            
        Returns:
            (pixel_x, pixel_y) 像素坐标
        """
        # 相对坐标
        rel_x = world_x - center_x
        rel_y = world_y - center_y
        
        # 转换为像素（Y轴翻转，图像坐标Y向下）
        px = int(self.offset_x + rel_x * self.PIXELS_PER_METER)
        py = int(self.offset_y - rel_y * self.PIXELS_PER_METER)
        
        return px, py

    def _draw_polygon_cv2(
        self, image: np.ndarray, world_corners: np.ndarray, channel: int,
        center_x: float, center_y: float, color: int = 255
    ):
        """使用OpenCV绘制多边形（快速）
        
        注意：只设置mask区域为1.0，不清空整个通道！
        """
        # 转换为像素坐标
        pixel_corners = []
        for corner in world_corners:
            px, py = self._world_to_pixel(corner[0], corner[1], center_x, center_y)
            pixel_corners.append([px, py])
        pixel_corners = np.array(pixel_corners, dtype=np.int32)

        # 创建mask并应用到指定通道（不清零，只设置多边形区域）
        mask = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_corners], 255)
        mask_bool = mask > 0

        if channel == 0:
            image[0] = np.where(mask_bool, 1.0, image[0])
        elif channel == 1:
            image[1] = np.where(mask_bool, 1.0, image[1])
        else:
            image[2] = np.where(mask_bool, 1.0, image[2])

    def _draw_vehicle_cv2(
        self, image: np.ndarray, state: VehicleState, channel: int,
        center_x: float, center_y: float, color: float = 1.0,
        mark_rear_axis: bool = False
    ):
        """使用OpenCV绘制车辆（快速）
        
        业内标准：state.x, state.y = 后轴中心位置
        车辆轮廓根据后轴中心 + 前悬/后悬/车长/车宽计算
        
        布局:
            车头 ──────────────┐
                              │ ← 前悬到前轴
            前轴 ───轴距─── 后轴(state.x, state.y)
                              │ ← 后悬
            车尾 ──────────────┘
        """
        cos_h = math.cos(state.heading)
        sin_h = math.sin(state.heading)
        
        # 车辆各关键点沿heading方向的距离（从后轴中心算起）
        rear_overhang = VEHICLE_CONFIG.rear_overhang       # 后轴到车尾
        front_overhang = VEHICLE_CONFIG.length - VEHICLE_CONFIG.rear_overhang  # 后轴到车头
        half_w = VEHICLE_CONFIG.width / 2
        
        # 车辆矩形四角点（相对于后轴中心的局部坐标）
        # 沿heading方向: [-rear_overhang, +front_overhang]
        # 垂直heading方向: [-half_w, +half_w]
        local_corners = np.array([
            [-rear_overhang, -half_w],   # 左后角
            [front_overhang, -half_w],   # 左前角
            [front_overhang, half_w],    # 右前角
            [-rear_overhang, half_w],    # 右后角
        ])
        
        # 旋转+平移到世界坐标
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        world_corners = local_corners @ rotation.T
        world_corners[:, 0] += state.x
        world_corners[:, 1] += state.y
        
        self._draw_polygon_cv2(image, world_corners, channel, center_x, center_y)
        
        # 标记后轴中心点：在车辆矩形上方画一个醒目的十字+箭头
        if mark_rear_axis:
            px, py = self._world_to_pixel(state.x, state.y, center_x, center_y)
            if 0 <= py < self.IMAGE_SIZE and 0 <= px < self.IMAGE_SIZE:
                # 1. 画十字标记后轴中心（比车辆矩形更醒目）
                cross_len = max(8, int(0.3 * self.PIXELS_PER_METER))
                cross_thickness = 2
                
                # 水平线
                cv2.line(image[channel], (px-cross_len, py), (px+cross_len, py), 1.0, cross_thickness)
                # 垂直线
                cv2.line(image[channel], (px, py-cross_len), (px, py+cross_len), 1.0, cross_thickness)
                
                # 2. 画中心圆点
                cv2.circle(image[channel], (px, py), max(3, int(0.08 * self.PIXELS_PER_METER)), 1.0, -1)

                # 3. 画朝向箭头（从后轴中心指向车头方向）
                arrow_len = max(10, int(0.6 * self.PIXELS_PER_METER))
                arrow_end_x = int(px + arrow_len * cos_h)
                arrow_end_y = int(py - arrow_len * sin_h)  # Y轴翻转
                
                arrow_end_x = max(0, min(self.IMAGE_SIZE - 1, arrow_end_x))
                arrow_end_y = max(0, min(self.IMAGE_SIZE - 1, arrow_end_y))
                
                cv2.arrowedLine(image[channel], (px, py), (arrow_end_x, arrow_end_y), 1.0, 2, tipLength=0.5)

    def render(
        self,
        state: VehicleState,
        target_state: VehicleState,
        obstacles: List[Rectangle]
    ) -> np.ndarray:
        """
        渲染场景
        
        Args:
            state: 当前车辆状态
            target_state: 目标状态
            obstacles: 障碍物列表
            
        Returns:
            图像数组 (3, 384, 384)，值范围 [0, 1]
        """
        # 创建空图像 (C, H, W)
        image = np.zeros((3, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)
        
        # 以车辆当前位置为视野中心
        center_x = state.x
        center_y = state.y
        
        # 绘制障碍物 (B通道)
        for obs in obstacles:
            corners = obs.get_corners()
            self._draw_polygon_cv2(image, corners, channel=2,
                                   center_x=center_x, center_y=center_y)
        
        # 绘制目标位置 (G通道)
        self._draw_vehicle_cv2(image, target_state, channel=1,
                               center_x=center_x, center_y=center_y,
                               mark_rear_axis=True)
        
        # 绘制当前车辆 (R通道)
        self._draw_vehicle_cv2(image, state, channel=0,
                               center_x=center_x, center_y=center_y,
                               mark_rear_axis=True)
        
        return image


def render_sample(state, target_state, obstacles, renderer=None) -> np.ndarray:
    """便捷函数：渲染单个样本"""
    if renderer is None:
        renderer = SceneRenderer()
    return renderer.render(state, target_state, obstacles)
