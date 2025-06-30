import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ExposureAnalyzer:
    """Image exposure analyzer
    
    Used to detect overexposure or underexposure in images
    """
    
    def __init__(self,
                 overexposure_threshold: float = 0.05,
                 underexposure_threshold: float = 0.05,
                 bright_pixel_threshold: int = 240,
                 dark_pixel_threshold: int = 15,
                 histogram_bins: int = 256):
        """Initialize exposure analyzer
        
        Args:
            overexposure_threshold: Overexposed pixel ratio threshold
            underexposure_threshold: Underexposed pixel ratio threshold
            bright_pixel_threshold: Bright pixel threshold
            dark_pixel_threshold: Dark pixel threshold
            histogram_bins: Number of histogram bins
        """
        self.overexposure_threshold = overexposure_threshold
        self.underexposure_threshold = underexposure_threshold
        self.bright_pixel_threshold = bright_pixel_threshold
        self.dark_pixel_threshold = dark_pixel_threshold
        self.histogram_bins = histogram_bins
        
    def analyze_exposure(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Analyze image exposure status
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (exposure status, metrics dictionary)
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
        
        # Calculate overexposed and underexposed pixel ratios
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
        
        # Determine exposure status
        exposure_status = 'normal'
        
        if bright_ratio > self.overexposure_threshold:
            exposure_status = 'overexposed'
        elif dark_ratio > self.underexposure_threshold:
            exposure_status = 'underexposed'
        elif mean_brightness > 220:  # Overall too bright
            exposure_status = 'overexposed'
        elif mean_brightness < 35:   # Overall too dark
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
        
        # Analyze saturation (saturation decreases with overexposure)
        s_mean = np.mean(s)
        s_std = np.std(s)
        
        # Low saturation with high brightness may indicate overexposure
        low_saturation_ratio = np.sum((s < 30) & (v > 220)) / (v.shape[0] * v.shape[1])
        
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
            # Grayscale
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
            Exposure quality score (0-1, 1 means best)
        """
        exposure_status, metrics = self.analyze_exposure(image)
        
        # Base score
        score = 1.0
        
        # Deduct points based on exposure status
        if exposure_status == 'overexposed':
            score -= 0.5
        elif exposure_status == 'underexposed':
            score -= 0.5
            
        # Deduct points based on brightness distribution
        mean_brightness = metrics['mean_brightness']
        ideal_brightness = 128
        brightness_penalty = abs(mean_brightness - ideal_brightness) / ideal_brightness
        score -= brightness_penalty * 0.3
        
        # Deduct points based on bright/dark pixel ratios
        score -= metrics['bright_pixel_ratio'] * 2
        score -= metrics['dark_pixel_ratio'] * 2
        
        # Limit to 0-1 range
        return max(0.0, min(1.0, score))