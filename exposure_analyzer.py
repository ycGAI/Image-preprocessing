import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ExposureAnalyzer:
    """Image exposure analyzer
    
    Detects overexposed or underexposed images
    """
    
    def __init__(self,
                 overexposure_threshold: float = 0.15,  # Relaxed from 0.05 to 0.15 (15%)
                 underexposure_threshold: float = 0.10,  # Relaxed from 0.05 to 0.10 (10%)
                 bright_pixel_threshold: int = 245,      # Increased from 240 to 245
                 dark_pixel_threshold: int = 10,         # Decreased from 15 to 10
                 histogram_bins: int = 256,
                 mean_brightness_overexposed: int = 235,  # New: mean brightness threshold for overexposure
                 mean_brightness_underexposed: int = 20): # New: mean brightness threshold for underexposure
        """Initialize exposure analyzer
        
        Args:
            overexposure_threshold: Ratio threshold for overexposed pixels
            underexposure_threshold: Ratio threshold for underexposed pixels
            bright_pixel_threshold: Threshold for bright pixels
            dark_pixel_threshold: Threshold for dark pixels
            histogram_bins: Number of histogram bins
            mean_brightness_overexposed: Mean brightness threshold for overexposure
            mean_brightness_underexposed: Mean brightness threshold for underexposure
        """
        self.overexposure_threshold = overexposure_threshold
        self.underexposure_threshold = underexposure_threshold
        self.bright_pixel_threshold = bright_pixel_threshold
        self.dark_pixel_threshold = dark_pixel_threshold
        self.histogram_bins = histogram_bins
        self.mean_brightness_overexposed = mean_brightness_overexposed
        self.mean_brightness_underexposed = mean_brightness_underexposed
        
    def analyze_exposure(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Analyze image exposure
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (exposure status, metrics dict)
            exposure status: 'normal', 'overexposed', 'underexposed'
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [self.histogram_bins], [0, 256])
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Calculate ratio of overexposed and underexposed pixels
        bright_pixels = np.sum(hist[self.bright_pixel_threshold:])
        dark_pixels = np.sum(hist[:self.dark_pixel_threshold])
        
        bright_ratio = bright_pixels / total_pixels
        dark_ratio = dark_pixels / total_pixels
        
        # Calculate mean brightness and standard deviation
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Calculate brightness distribution skewness
        hist_normalized = hist.flatten() / total_pixels
        bins = np.arange(self.histogram_bins)
        mean_bin = np.sum(bins * hist_normalized)
        
        # Third central moment (skewness)
        skewness = np.sum(((bins - mean_bin) ** 3) * hist_normalized)
        
        # Calculate standard deviation in bright regions (new feature)
        bright_region_mask = gray > self.bright_pixel_threshold
        if np.any(bright_region_mask):
            bright_region_std = np.std(gray[bright_region_mask])
        else:
            bright_region_std = 0
        
        # Determine exposure status - more relaxed judgment logic
        exposure_status = 'normal'
        
        # Overexposure judgment: multiple conditions must be met
        overexposure_score = 0
        
        if bright_ratio > self.overexposure_threshold:
            overexposure_score += 2  # Main indicator, higher weight
            
        if mean_brightness > self.mean_brightness_overexposed:
            overexposure_score += 2
        elif mean_brightness > 200:  # Moderately bright
            overexposure_score += 1
            
        # If bright regions have detail (high std), not completely overexposed
        if bright_region_std > 20:
            overexposure_score -= 1
            
        # Only consider overexposed if score >= 3
        if overexposure_score >= 3:
            exposure_status = 'overexposed'
            
        # Underexposure judgment
        elif dark_ratio > self.underexposure_threshold:
            exposure_status = 'underexposed'
        elif mean_brightness < self.mean_brightness_underexposed:
            exposure_status = 'underexposed'
            
        metrics = {
            'bright_pixel_ratio': float(bright_ratio),
            'dark_pixel_ratio': float(dark_ratio),
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'brightness_skewness': float(skewness),
            'exposure_status': exposure_status,
            'overexposure_score': overexposure_score,
            'bright_region_std': float(bright_region_std)
        }
        
        return exposure_status, metrics
    
    def analyze_hsv_exposure(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze exposure using HSV color space
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            HSV analysis metrics
        """
        # Convert to HSV space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Analyze value channel
        v_mean = np.mean(v)
        v_std = np.std(v)
        
        # Analyze saturation (saturation decreases when overexposed)
        s_mean = np.mean(s)
        s_std = np.std(s)
        
        # Low saturation with high brightness might indicate overexposure
        # Relaxed condition: only consider problematic when saturation is very low and brightness very high
        low_saturation_ratio = np.sum((s < 20) & (v > 250)) / (v.shape[0] * v.shape[1])
        
        return {
            'v_channel_mean': float(v_mean),
            'v_channel_std': float(v_std),
            's_channel_mean': float(s_mean),
            's_channel_std': float(s_std),
            'low_saturation_bright_ratio': float(low_saturation_ratio)
        }
    
    def detect_clipping(self, image: np.ndarray) -> Dict[str, float]:
        """Detect highlight and shadow clipping
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Clipping detection metrics
        """
        # Detect clipping in each channel
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
            # Grayscale image
            highlight_clipping = np.sum(image >= 255) / image.size
            shadow_clipping = np.sum(image <= 0) / image.size
            
            clipping_metrics['highlight_clipping'] = float(highlight_clipping)
            clipping_metrics['shadow_clipping'] = float(shadow_clipping)
            
        return clipping_metrics
    
    def get_exposure_score(self, image: np.ndarray) -> float:
        """Calculate exposure quality score
        
        Args:
            image: Input image
            
        Returns:
            Exposure quality score (0-1, 1 is best)
        """
        exposure_status, metrics = self.analyze_exposure(image)
        
        # Base score
        score = 1.0
        
        # Deduct points based on exposure status
        if exposure_status == 'overexposed':
            score -= 0.3  # Reduced penalty from 0.5 to 0.3
        elif exposure_status == 'underexposed':
            score -= 0.4  # Slightly less penalty for underexposure
            
        # Deduct points based on brightness distribution
        mean_brightness = metrics['mean_brightness']
        ideal_brightness = 128
        brightness_penalty = abs(mean_brightness - ideal_brightness) / ideal_brightness
        score -= brightness_penalty * 0.2  # Reduced from 0.3 to 0.2
        
        # Deduct points based on bright/dark pixel ratios
        score -= metrics['bright_pixel_ratio'] * 1.5  # Reduced from 2 to 1.5
        score -= metrics['dark_pixel_ratio'] * 1.5   # Reduced from 2 to 1.5
        
        # Constrain to 0-1 range
        return max(0.0, min(1.0, score))