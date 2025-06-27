import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class WorkAreaDetector:
    """工作区域检测器
    
    用于检测图像是否在农场工作区域内（包含植物和土壤）
    """
    
    def __init__(self,
                 green_threshold: float = 0.15,
                 brown_threshold: float = 0.10,
                 vegetation_index_threshold: float = 0.1,
                 texture_threshold: float = 20.0,
                 min_valid_area: float = 0.3):
        """初始化工作区域检测器
        
        Args:
            green_threshold: 绿色区域最小比例
            brown_threshold: 棕色/土壤区域最小比例
            vegetation_index_threshold: 植被指数阈值
            texture_threshold: 纹理复杂度阈值
            min_valid_area: 最小有效区域比例
        """
        self.green_threshold = green_threshold
        self.brown_threshold = brown_threshold
        self.vegetation_index_threshold = vegetation_index_threshold
        self.texture_threshold = texture_threshold
        self.min_valid_area = min_valid_area
        
    def detect_green_vegetation(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """检测绿色植被区域
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            (绿色区域比例, 绿色掩码)
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义绿色范围
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        # 创建绿色掩码
        mask_green = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        
        # 计算绿色区域比例
        green_ratio = np.sum(mask_green > 0) / (image.shape[0] * image.shape[1])
        
        return float(green_ratio), mask_green
    
    def detect_soil_area(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """检测土壤区域
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            (土壤区域比例, 土壤掩码)
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义棕色/土壤颜色范围
        lower_brown = np.array([10, 20, 20])
        upper_brown = np.array([25, 255, 200])
        
        # 创建土壤掩码
        mask_soil = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # 额外检测暗色土壤
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 100, 80])
        mask_dark_soil = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # 合并掩码
        mask_soil = cv2.bitwise_or(mask_soil, mask_dark_soil)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask_soil = cv2.morphologyEx(mask_soil, cv2.MORPH_CLOSE, kernel)
        mask_soil = cv2.morphologyEx(mask_soil, cv2.MORPH_OPEN, kernel)
        
        # 计算土壤区域比例
        soil_ratio = np.sum(mask_soil > 0) / (image.shape[0] * image.shape[1])
        
        return float(soil_ratio), mask_soil
    
    def calculate_vegetation_index(self, image: np.ndarray) -> float:
        """计算植被指数 (简化的NDVI替代)
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            植被指数
        """
        # 分离通道
        b, g, r = cv2.split(image.astype(np.float32))
        
        # 避免除零
        epsilon = 1e-10
        
        # 计算归一化差异植被指数的简化版本
        # 使用绿色和红色通道
        vegetation_index = np.mean((g - r) / (g + r + epsilon))
        
        # 额外的绿色指数
        excess_green = 2 * g - r - b
        eg_index = np.mean(excess_green) / 255.0
        
        # 综合指数
        combined_index = (vegetation_index + eg_index) / 2
        
        return float(combined_index)
    
    def analyze_texture(self, image: np.ndarray) -> float:
        """分析图像纹理复杂度
        
        Args:
            image: 输入图像
            
        Returns:
            纹理复杂度分数
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算Sobel梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # 纹理复杂度为梯度的标准差
        texture_complexity = np.std(gradient_magnitude)
        
        return float(texture_complexity)
    
    def detect_farm_elements(self, image: np.ndarray) -> Dict[str, float]:
        """检测农场相关元素
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果字典
        """
        # 检测植物行列模式
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Hough变换检测直线（农作物行）
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=100, maxLineGap=10)
        
        line_count = 0 if lines is None else len(lines)
        
        # 检测规则模式（使用FFT）
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 计算频谱的规则性
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # 分析中心区域外的频谱能量（规则模式会在频谱中产生峰值）
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 0
        
        pattern_energy = np.sum(magnitude_spectrum * mask) / np.sum(magnitude_spectrum)
        
        return {
            'line_count': float(line_count),
            'pattern_regularity': float(pattern_energy)
        }
    
    def is_in_work_area(self, image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """判断图像是否在工作区域内
        
        Args:
            image: 输入图像
            
        Returns:
            (是否在工作区域, 检测指标)
        """
        metrics = {}
        
        # 检测绿色植被
        green_ratio, green_mask = self.detect_green_vegetation(image)
        metrics['green_vegetation_ratio'] = green_ratio
        
        # 检测土壤
        soil_ratio, soil_mask = self.detect_soil_area(image)
        metrics['soil_ratio'] = soil_ratio
        
        # 计算植被指数
        vegetation_index = self.calculate_vegetation_index(image)
        metrics['vegetation_index'] = vegetation_index
        
        # 分析纹理
        texture_complexity = self.analyze_texture(image)
        metrics['texture_complexity'] = texture_complexity
        
        # 检测农场元素
        farm_elements = self.detect_farm_elements(image)
        metrics.update(farm_elements)
        
        # 计算总的自然元素比例
        natural_area_ratio = green_ratio + soil_ratio
        metrics['natural_area_ratio'] = float(natural_area_ratio)
        
        # 综合判断
        in_work_area = True
        reasons = []
        
        # 检查是否有足够的植被或土壤
        if green_ratio < self.green_threshold and soil_ratio < self.brown_threshold:
            in_work_area = False
            reasons.append("insufficient_vegetation_or_soil")
            
        # 检查植被指数
        if vegetation_index < self.vegetation_index_threshold and green_ratio < 0.1:
            in_work_area = False
            reasons.append("low_vegetation_index")
            
        # 检查自然区域总比例
        if natural_area_ratio < self.min_valid_area:
            in_work_area = False
            reasons.append("insufficient_natural_area")
            
        # 检查纹理（太低可能是均匀背景，如天空）
        if texture_complexity < self.texture_threshold and natural_area_ratio < 0.2:
            in_work_area = False
            reasons.append("uniform_background")
            
        metrics['in_work_area'] = in_work_area
        metrics['out_of_area_reasons'] = reasons
        
        return in_work_area, metrics
    
    def visualize_detection(self, image: np.ndarray) -> np.ndarray:
        """可视化检测结果
        
        Args:
            image: 输入图像
            
        Returns:
            可视化结果图像
        """
        # 检测各个区域
        green_ratio, green_mask = self.detect_green_vegetation(image)
        soil_ratio, soil_mask = self.detect_soil_area(image)
        
        # 创建彩色掩码
        result = image.copy()
        overlay = np.zeros_like(image)
        
        # 绿色区域标记为绿色
        overlay[green_mask > 0] = [0, 255, 0]
        
        # 土壤区域标记为棕色
        overlay[soil_mask > 0] = [42, 42, 165]  # BGR格式的棕色
        
        # 混合原图和掩码
        alpha = 0.3
        result = cv2.addWeighted(result, 1-alpha, overlay, alpha, 0)
        
        # 添加文字信息
        in_area, metrics = self.is_in_work_area(image)
        status_text = "In Work Area" if in_area else "Out of Work Area"
        color = (0, 255, 0) if in_area else (0, 0, 255)
        
        cv2.putText(result, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(result, f"Green: {green_ratio:.1%}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(result, f"Soil: {soil_ratio:.1%}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (42, 42, 165), 1)
        
        return result