import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ExposureAnalyzer:
    """图像曝光分析器
    
    用于检测图像是否过曝或欠曝光
    """
    
    def __init__(self,
                 overexposure_threshold: float = 0.05,
                 underexposure_threshold: float = 0.05,
                 bright_pixel_threshold: int = 240,
                 dark_pixel_threshold: int = 15,
                 histogram_bins: int = 256):
        """初始化曝光分析器
        
        Args:
            overexposure_threshold: 过曝像素比例阈值
            underexposure_threshold: 欠曝像素比例阈值
            bright_pixel_threshold: 亮像素阈值
            dark_pixel_threshold: 暗像素阈值
            histogram_bins: 直方图bins数量
        """
        self.overexposure_threshold = overexposure_threshold
        self.underexposure_threshold = underexposure_threshold
        self.bright_pixel_threshold = bright_pixel_threshold
        self.dark_pixel_threshold = dark_pixel_threshold
        self.histogram_bins = histogram_bins
        
    def analyze_exposure(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """分析图像曝光情况
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            (曝光状态, 指标字典)
            曝光状态: 'normal', 'overexposed', 'underexposed'
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [self.histogram_bins], [0, 256])
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # 计算过曝和欠曝像素比例
        bright_pixels = np.sum(hist[self.bright_pixel_threshold:])
        dark_pixels = np.sum(hist[:self.dark_pixel_threshold])
        
        bright_ratio = bright_pixels / total_pixels
        dark_ratio = dark_pixels / total_pixels
        
        # 计算平均亮度和标准差
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # 计算亮度分布的偏度
        hist_normalized = hist.flatten() / total_pixels
        bins = np.arange(self.histogram_bins)
        mean_bin = np.sum(bins * hist_normalized)
        
        # 三阶中心矩（偏度）
        skewness = np.sum(((bins - mean_bin) ** 3) * hist_normalized)
        
        # 判断曝光状态
        exposure_status = 'normal'
        
        if bright_ratio > self.overexposure_threshold:
            exposure_status = 'overexposed'
        elif dark_ratio > self.underexposure_threshold:
            exposure_status = 'underexposed'
        elif mean_brightness > 220:  # 整体过亮
            exposure_status = 'overexposed'
        elif mean_brightness < 35:   # 整体过暗
            exposure_status = 'underexposed'
            
        metrics = {
            'bright_pixel_ratio': float(bright_ratio),
            'dark_pixel_ratio': float(dark_ratio),
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'brightness_skewness': float(skewness),
            'exposure_status': exposure_status
        }
        
        return exposure_status, metrics
    
    def analyze_hsv_exposure(self, image: np.ndarray) -> Dict[str, float]:
        """使用HSV色彩空间分析曝光
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            HSV分析指标
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 分析明度通道
        v_mean = np.mean(v)
        v_std = np.std(v)
        
        # 分析饱和度（过曝时饱和度会降低）
        s_mean = np.mean(s)
        s_std = np.std(s)
        
        # 低饱和度高亮度可能表示过曝
        low_saturation_ratio = np.sum((s < 30) & (v > 220)) / (v.shape[0] * v.shape[1])
        
        return {
            'v_channel_mean': float(v_mean),
            'v_channel_std': float(v_std),
            's_channel_mean': float(s_mean),
            's_channel_std': float(s_std),
            'low_saturation_bright_ratio': float(low_saturation_ratio)
        }
    
    def detect_clipping(self, image: np.ndarray) -> Dict[str, float]:
        """检测高光和暗部裁剪
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            裁剪检测指标
        """
        # 检测每个通道的裁剪
        clipping_metrics = {}
        
        if len(image.shape) == 3:
            channels = ['blue', 'green', 'red']
            for i, channel_name in enumerate(channels):
                channel = image[:, :, i]
                highlight_clipping = np.sum(channel >= 255) / channel.size
                shadow_clipping = np.sum(channel <= 0) / channel.size
                
                clipping_metrics[f'{channel_name}_highlight_clipping'] = float(highlight_clipping)
                clipping_metrics[f'{channel_name}_shadow_clipping'] = float(shadow_clipping)
        else:
            # 灰度图
            highlight_clipping = np.sum(image >= 255) / image.size
            shadow_clipping = np.sum(image <= 0) / image.size
            
            clipping_metrics['highlight_clipping'] = float(highlight_clipping)
            clipping_metrics['shadow_clipping'] = float(shadow_clipping)
            
        return clipping_metrics
    
    def get_exposure_score(self, image: np.ndarray) -> float:
        """计算曝光质量分数
        
        Args:
            image: 输入图像
            
        Returns:
            曝光质量分数 (0-1, 1表示最佳)
        """
        exposure_status, metrics = self.analyze_exposure(image)
        
        # 基础分数
        score = 1.0
        
        # 根据曝光状态扣分
        if exposure_status == 'overexposed':
            score -= 0.5
        elif exposure_status == 'underexposed':
            score -= 0.5
            
        # 根据亮度分布扣分
        mean_brightness = metrics['mean_brightness']
        ideal_brightness = 128
        brightness_penalty = abs(mean_brightness - ideal_brightness) / ideal_brightness
        score -= brightness_penalty * 0.3
        
        # 根据明暗像素比例扣分
        score -= metrics['bright_pixel_ratio'] * 2
        score -= metrics['dark_pixel_ratio'] * 2
        
        # 限制在0-1范围内
        return max(0.0, min(1.0, score))