import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class WorkAreaDetector:
    """工作区域检测器 - 简化版
    
    通过颜色判断：
    - 工作区域：大部分是土地（棕色/暗色），只有少量植物
    - 非工作区域：大部分是草地（绿色）
    """
    
    def __init__(self,
                 grass_threshold: float = 0.5,      # 草地判定阈值（绿色比例>50%认为是草地）
                 soil_min_threshold: float = 0.3,   # 工作区域最小土壤比例
                 green_max_threshold: float = 0.3): # 工作区域最大绿色比例
        """初始化工作区域检测器
        
        Args:
            grass_threshold: 绿色比例超过此值判定为草地（非工作区域）
            soil_min_threshold: 土壤比例超过此值可能是工作区域
            green_max_threshold: 工作区域的绿色比例不应超过此值
        """
        self.grass_threshold = grass_threshold
        self.soil_min_threshold = soil_min_threshold
        self.green_max_threshold = green_max_threshold
    
    def detect_green_grass(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """检测绿色草地区域
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            (绿色区域比例, 绿色掩码)
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义草地绿色范围（比较宽泛）
        # 草地的绿色通常比较鲜艳
        lower_green = np.array([25, 30, 30])    # 色相25-85覆盖各种绿色
        upper_green = np.array([85, 255, 255])
        
        # 创建绿色掩码
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        
        # 计算绿色区域比例
        green_ratio = np.sum(mask_green > 0) / (image.shape[0] * image.shape[1])
        
        return float(green_ratio), mask_green
    
    def detect_soil(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """检测土壤区域（棕色和暗色区域）
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            (土壤区域比例, 土壤掩码)
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义多种土壤颜色范围
        masks = []
        
        # 1. 棕色土壤（色相10-25）
        lower_brown = np.array([10, 20, 20])
        upper_brown = np.array([25, 255, 180])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        masks.append(mask_brown)
        
        # 2. 暗色土壤（低亮度）
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])  # 亮度<80
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        masks.append(mask_dark)
        
        # 3. 灰色土壤（低饱和度）
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([180, 30, 120])  # 饱和度<30
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        masks.append(mask_gray)
        
        # 合并所有土壤掩码
        mask_soil = masks[0]
        for mask in masks[1:]:
            mask_soil = cv2.bitwise_or(mask_soil, mask)
        
        # 去除与绿色重叠的部分
        _, mask_green = self.detect_green_grass(image)
        mask_soil = cv2.bitwise_and(mask_soil, cv2.bitwise_not(mask_green))
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask_soil = cv2.morphologyEx(mask_soil, cv2.MORPH_OPEN, kernel)
        mask_soil = cv2.morphologyEx(mask_soil, cv2.MORPH_CLOSE, kernel)
        
        # 计算土壤区域比例
        soil_ratio = np.sum(mask_soil > 0) / (image.shape[0] * image.shape[1])
        
        return float(soil_ratio), mask_soil
    
    def is_in_work_area(self, image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """判断图像是否在工作区域内
        
        判断逻辑：
        1. 如果绿色比例>50%，判定为草地（非工作区域）
        2. 如果土壤比例>30%且绿色比例<30%，判定为工作区域
        3. 其他情况根据绿色和土壤的比例综合判断
        
        Args:
            image: 输入图像
            
        Returns:
            (是否在工作区域, 检测指标)
        """
        # 检测绿色草地
        green_ratio, green_mask = self.detect_green_grass(image)
        
        # 检测土壤
        soil_ratio, soil_mask = self.detect_soil(image)
        
        # 计算其他区域（可能是天空、道路等）
        other_ratio = 1.0 - green_ratio - soil_ratio
        
        # 构建指标
        metrics = {
            'green_ratio': green_ratio,
            'soil_ratio': soil_ratio,
            'other_ratio': float(other_ratio),
            'green_soil_ratio': green_ratio / (soil_ratio + 0.001)  # 避免除零
        }
        
        # 判断逻辑
        in_work_area = True
        reasons = []
        
        # 规则1：绿色太多，判定为草地
        if green_ratio > self.grass_threshold:
            in_work_area = False
            reasons.append(f"too_much_green ({green_ratio:.1%})")
        
        # 规则2：土壤充足且绿色较少，判定为工作区域
        elif soil_ratio > self.soil_min_threshold and green_ratio < self.green_max_threshold:
            in_work_area = True
            # 这是典型的工作区域
        
        # 规则3：土壤太少
        elif soil_ratio < 0.1:
            in_work_area = False
            reasons.append(f"insufficient_soil ({soil_ratio:.1%})")
        
        # 规则4：绿色和土壤都很少，可能是其他区域（天空、建筑等）
        elif green_ratio < 0.1 and soil_ratio < 0.2:
            in_work_area = False
            reasons.append("not_farmland")
        
        metrics['in_work_area'] = in_work_area
        metrics['reasons'] = reasons
        
        return in_work_area, metrics
    
    def visualize_detection(self, image: np.ndarray) -> np.ndarray:
        """可视化检测结果
        
        Args:
            image: 输入图像
            
        Returns:
            可视化结果图像
        """
        # 检测各个区域
        green_ratio, green_mask = self.detect_green_grass(image)
        soil_ratio, soil_mask = self.detect_soil(image)
        
        # 创建彩色掩码
        result = image.copy()
        overlay = np.zeros_like(image)
        
        # 绿色区域标记为亮绿色
        overlay[green_mask > 0] = [0, 255, 0]
        
        # 土壤区域标记为棕色
        overlay[soil_mask > 0] = [42, 42, 165]  # BGR格式的棕色
        
        # 混合原图和掩码
        alpha = 0.3
        result = cv2.addWeighted(result, 1-alpha, overlay, alpha, 0)
        
        # 添加文字信息
        in_area, metrics = self.is_in_work_area(image)
        status_text = "Work Area" if in_area else "Non-Work Area"
        color = (0, 255, 0) if in_area else (0, 0, 255)
        
        # 背景框
        cv2.rectangle(result, (5, 5), (300, 110), (0, 0, 0), -1)
        cv2.rectangle(result, (5, 5), (300, 110), color, 2)
        
        # 文字信息
        cv2.putText(result, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(result, f"Green: {green_ratio:.1%}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(result, f"Soil: {soil_ratio:.1%}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (42, 42, 165), 1)
        cv2.putText(result, f"Other: {metrics['other_ratio']:.1%}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result
    
    def get_color_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """获取详细的颜色统计信息
        
        Args:
            image: 输入图像
            
        Returns:
            颜色统计信息
        """
        in_area, metrics = self.is_in_work_area(image)
        
        # 添加更多统计信息
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 平均色相、饱和度、亮度
        metrics['mean_hue'] = float(np.mean(hsv[:, :, 0]))
        metrics['mean_saturation'] = float(np.mean(hsv[:, :, 1]))
        metrics['mean_value'] = float(np.mean(hsv[:, :, 2]))
        
        # 主色调分析
        hue_hist, _ = np.histogram(hsv[:, :, 0], bins=18, range=(0, 180))
        dominant_hue_bin = np.argmax(hue_hist)
        metrics['dominant_hue'] = float(dominant_hue_bin * 10)  # 转换为角度
        
        return metrics