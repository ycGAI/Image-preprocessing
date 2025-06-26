import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class OverexposureDetector:
    """图像过曝检测器
    
    使用多种算法检测图像是否存在过曝现象
    """
    
    def __init__(self,
                 highlight_threshold: float = 0.05,  # 高光像素比例阈值
                 brightness_threshold: float = 230.0,  # 亮度阈值
                 clipping_threshold: float = 250.0,  # 剪切检测阈值
                 local_overexposure_threshold: float = 0.1,  # 局部过曝比例阈值
                 saturation_loss_threshold: float = 0.03):  # 饱和度损失阈值
        """初始化过曝检测器
        
        Args:
            highlight_threshold: 高光像素占比阈值（0-1）
            brightness_threshold: 平均亮度阈值（0-255）
            clipping_threshold: 高光剪切检测阈值（0-255）
            local_overexposure_threshold: 局部过曝区域占比阈值（0-1）
            saturation_loss_threshold: 饱和度损失比例阈值（0-1）
        """
        self.thresholds = {
            'highlight_ratio': highlight_threshold,
            'brightness': brightness_threshold,
            'clipping': clipping_threshold,
            'local_overexposure': local_overexposure_threshold,
            'saturation_loss': saturation_loss_threshold
        }
        
    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        """将图像转换为灰度图像"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def highlight_ratio_analysis(self, image: np.ndarray) -> Tuple[float, bool]:
        """分析高光像素比例
        
        Args:
            image: 输入图像
            
        Returns:
            (高光像素比例, 是否过曝)
        """
        gray = self._convert_to_gray(image)
        
        # 计算高光像素（>240）的比例
        highlight_pixels = np.sum(gray > 240)
        total_pixels = gray.size
        highlight_ratio = highlight_pixels / total_pixels
        
        is_overexposed = highlight_ratio > self.thresholds['highlight_ratio']
        
        return highlight_ratio, is_overexposed
    
    def brightness_analysis(self, image: np.ndarray) -> Tuple[float, bool]:
        """分析平均亮度
        
        Args:
            image: 输入图像
            
        Returns:
            (平均亮度, 是否过曝)
        """
        # 转换到LAB色彩空间，L通道代表亮度
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image
            
        mean_brightness = np.mean(l_channel)
        
        # LAB的L通道范围是0-100，需要转换到0-255范围
        mean_brightness_255 = mean_brightness * 255 / 100
        
        is_overexposed = mean_brightness_255 > self.thresholds['brightness']
        
        return mean_brightness_255, is_overexposed
    
    def clipping_detection(self, image: np.ndarray) -> Tuple[float, bool]:
        """检测高光剪切
        
        Args:
            image: 输入图像
            
        Returns:
            (剪切像素比例, 是否过曝)
        """
        # 检测接近最大值的像素
        if len(image.shape) == 3:
            # 对于彩色图像，检查任一通道是否剪切
            clipped_pixels = np.sum(np.any(image >= self.thresholds['clipping'], axis=2))
        else:
            clipped_pixels = np.sum(image >= self.thresholds['clipping'])
            
        total_pixels = image.shape[0] * image.shape[1]
        clipping_ratio = clipped_pixels / total_pixels
        
        # 如果超过1%的像素被剪切，认为过曝
        is_overexposed = clipping_ratio > 0.01
        
        return clipping_ratio, is_overexposed
    
    def local_overexposure_detection(self, image: np.ndarray, window_size: int = 64) -> Tuple[float, bool]:
        """检测局部过曝区域
        
        Args:
            image: 输入图像
            window_size: 滑动窗口大小
            
        Returns:
            (过曝区域比例, 是否过曝)
        """
        gray = self._convert_to_gray(image)
        height, width = gray.shape
        
        overexposed_regions = 0
        total_regions = 0
        
        # 滑动窗口检测
        for y in range(0, height - window_size, window_size // 2):
            for x in range(0, width - window_size, window_size // 2):
                window = gray[y:y+window_size, x:x+window_size]
                
                # 检查窗口内的平均亮度和高亮像素比例
                window_mean = np.mean(window)
                highlight_ratio = np.sum(window > 240) / window.size
                
                if window_mean > 230 or highlight_ratio > 0.3:
                    overexposed_regions += 1
                    
                total_regions += 1
        
        overexposure_ratio = overexposed_regions / total_regions if total_regions > 0 else 0
        is_overexposed = overexposure_ratio > self.thresholds['local_overexposure']
        
        return overexposure_ratio, is_overexposed
    
    def saturation_loss_detection(self, image: np.ndarray) -> Tuple[float, bool]:
        """检测由于过曝导致的饱和度损失
        
        Args:
            image: 输入图像
            
        Returns:
            (饱和度损失比例, 是否过曝)
        """
        if len(image.shape) != 3:
            # 灰度图像无法检测饱和度
            return 0.0, False
            
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 找出高亮度但低饱和度的像素（可能是过曝区域）
        high_value_mask = v > 240
        low_saturation_mask = s < 30
        
        # 过曝区域：高亮度且低饱和度
        overexposed_mask = high_value_mask & low_saturation_mask
        
        saturation_loss_ratio = np.sum(overexposed_mask) / overexposed_mask.size
        is_overexposed = saturation_loss_ratio > self.thresholds['saturation_loss']
        
        return saturation_loss_ratio, is_overexposed
    
    def histogram_peak_analysis(self, image: np.ndarray) -> Tuple[float, bool]:
        """分析直方图峰值
        
        Args:
            image: 输入图像
            
        Returns:
            (高端峰值强度, 是否过曝)
        """
        gray = self._convert_to_gray(image)
        
        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # 归一化
        
        # 分析高端（240-255）的累积概率
        high_end_ratio = np.sum(hist[240:])
        
        # 检测是否有明显的高端峰值
        peak_threshold = 0.05  # 5%以上的像素集中在高端
        is_overexposed = high_end_ratio > peak_threshold
        
        return high_end_ratio, is_overexposed
    
    def calculate_all_metrics(self, image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """计算所有过曝检测指标
        
        Args:
            image: 输入图像
            
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 高光比例分析
        highlight_ratio, highlight_overexposed = self.highlight_ratio_analysis(image)
        metrics['highlight_ratio'] = {
            'value': float(highlight_ratio),
            'threshold': self.thresholds['highlight_ratio'],
            'is_overexposed': highlight_overexposed
        }
        
        # 亮度分析
        brightness, brightness_overexposed = self.brightness_analysis(image)
        metrics['brightness'] = {
            'value': float(brightness),
            'threshold': self.thresholds['brightness'],
            'is_overexposed': brightness_overexposed
        }
        
        # 剪切检测
        clipping_ratio, clipping_overexposed = self.clipping_detection(image)
        metrics['clipping'] = {
            'value': float(clipping_ratio),
            'threshold': 0.01,  # 1%
            'is_overexposed': clipping_overexposed
        }
        
        # 局部过曝检测
        local_ratio, local_overexposed = self.local_overexposure_detection(image)
        metrics['local_overexposure'] = {
            'value': float(local_ratio),
            'threshold': self.thresholds['local_overexposure'],
            'is_overexposed': local_overexposed
        }
        
        # 饱和度损失检测
        saturation_loss, saturation_overexposed = self.saturation_loss_detection(image)
        metrics['saturation_loss'] = {
            'value': float(saturation_loss),
            'threshold': self.thresholds['saturation_loss'],
            'is_overexposed': saturation_overexposed
        }
        
        # 直方图峰值分析
        histogram_peak, histogram_overexposed = self.histogram_peak_analysis(image)
        metrics['histogram_peak'] = {
            'value': float(histogram_peak),
            'threshold': 0.05,  # 5%
            'is_overexposed': histogram_overexposed
        }
        
        return metrics
    
    def detect_overexposure(self, image: np.ndarray, mode: str = 'ensemble') -> Tuple[bool, Dict]:
        """检测图像是否过曝
        
        Args:
            image: 输入图像
            mode: 检测模式 ('ensemble'=集成判断, 'strict'=严格模式, 'loose'=宽松模式)
            
        Returns:
            (是否过曝, 详细指标)
        """
        metrics = self.calculate_all_metrics(image)
        
        # 统计各指标的过曝判断
        overexposed_count = sum(1 for m in metrics.values() if m['is_overexposed'])
        total_metrics = len(metrics)
        
        if mode == 'ensemble':
            # 集成模式：超过半数指标认为过曝
            is_overexposed = overexposed_count > total_metrics / 2
        elif mode == 'strict':
            # 严格模式：任一指标认为过曝
            is_overexposed = overexposed_count > 0
        elif mode == 'loose':
            # 宽松模式：至少3个指标认为过曝
            is_overexposed = overexposed_count >= 3
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        # 计算过曝严重程度（0-1）
        severity = self._calculate_severity(metrics)
        
        result = {
            'is_overexposed': is_overexposed,
            'overexposed_metrics_count': overexposed_count,
            'total_metrics': total_metrics,
            'severity': severity,
            'mode': mode,
            'metrics': metrics
        }
        
        return is_overexposed, result
    
    def _calculate_severity(self, metrics: Dict) -> float:
        """计算过曝严重程度
        
        Args:
            metrics: 各项指标
            
        Returns:
            严重程度（0-1）
        """
        severity_scores = []
        
        # 根据各指标计算严重程度分数
        if 'highlight_ratio' in metrics:
            ratio = metrics['highlight_ratio']['value']
            threshold = metrics['highlight_ratio']['threshold']
            if ratio > threshold:
                severity_scores.append(min(ratio / (threshold * 3), 1.0))
        
        if 'brightness' in metrics:
            brightness = metrics['brightness']['value']
            if brightness > 230:
                severity_scores.append(min((brightness - 230) / 25, 1.0))
        
        if 'clipping' in metrics:
            clipping = metrics['clipping']['value']
            if clipping > 0.01:
                severity_scores.append(min(clipping / 0.05, 1.0))
        
        if 'local_overexposure' in metrics:
            local = metrics['local_overexposure']['value']
            threshold = metrics['local_overexposure']['threshold']
            if local > threshold:
                severity_scores.append(min(local / (threshold * 3), 1.0))
        
        # 返回平均严重程度
        return np.mean(severity_scores) if severity_scores else 0.0
    
    def get_overexposed_regions(self, image: np.ndarray, threshold: int = 240) -> np.ndarray:
        """获取过曝区域的掩码
        
        Args:
            image: 输入图像
            threshold: 亮度阈值
            
        Returns:
            过曝区域掩码（二值图像）
        """
        gray = self._convert_to_gray(image)
        
        # 创建过曝区域掩码
        overexposed_mask = gray > threshold
        
        # 应用形态学操作去除噪点
        kernel = np.ones((5, 5), np.uint8)
        overexposed_mask = cv2.morphologyEx(overexposed_mask.astype(np.uint8) * 255, 
                                           cv2.MORPH_OPEN, kernel)
        overexposed_mask = cv2.morphologyEx(overexposed_mask, cv2.MORPH_CLOSE, kernel)
        
        return overexposed_mask
    
    def visualize_overexposure(self, image: np.ndarray) -> np.ndarray:
        """可视化过曝区域
        
        Args:
            image: 输入图像
            
        Returns:
            标记了过曝区域的图像
        """
        # 获取过曝区域掩码
        mask = self.get_overexposed_regions(image)
        
        # 创建输出图像
        output = image.copy()
        
        # 将过曝区域标记为红色
        if len(output.shape) == 3:
            output[mask > 0] = [0, 0, 255]  # BGR格式，红色
        else:
            output[mask > 0] = 255
        
        # 添加半透明效果
        output = cv2.addWeighted(image, 0.7, output, 0.3, 0)
        
        return output
    
    def update_thresholds(self, **kwargs):
        """更新检测阈值"""
        for key, value in kwargs.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                logger.info(f"更新阈值 {key}: {value}")
            else:
                logger.warning(f"未知的阈值参数: {key}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """获取当前阈值设置"""
        return self.thresholds.copy()
    
    def analyze_batch(self, images: List[np.ndarray]) -> Dict:
        """批量分析图像过曝情况
        
        Args:
            images: 图像列表
            
        Returns:
            批量分析结果
        """
        results = []
        overexposed_count = 0
        severity_scores = []
        
        for i, image in enumerate(images):
            is_overexposed, details = self.detect_overexposure(image)
            results.append({
                'index': i,
                'is_overexposed': is_overexposed,
                'severity': details['severity']
            })
            
            if is_overexposed:
                overexposed_count += 1
            severity_scores.append(details['severity'])
        
        return {
            'total_images': len(images),
            'overexposed_count': overexposed_count,
            'overexposure_ratio': overexposed_count / len(images) if images else 0,
            'average_severity': np.mean(severity_scores) if severity_scores else 0,
            'max_severity': np.max(severity_scores) if severity_scores else 0,
            'results': results
        }