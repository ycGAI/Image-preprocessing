import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class WorkAreaDetector:
    """Work area detector - Improved version
    
    Determination by color:
    - Work area: Mostly soil (brown/gray/dark), with minimal plants
    - Non-work area: Mostly grass (green)
    """
    
    def __init__(self,
                 grass_threshold: float = 0.5,      # Grass threshold (green ratio > 50% is considered grass)
                 soil_min_threshold: float = 0.3,   # Minimum soil ratio for work area
                 green_max_threshold: float = 0.3): # Maximum green ratio for work area
        """Initialize work area detector
        
        Args:
            grass_threshold: Green ratio exceeding this value is considered grass (non-work area)
            soil_min_threshold: Soil ratio exceeding this value may be work area
            green_max_threshold: Green ratio in work area should not exceed this value
        """
        self.grass_threshold = grass_threshold
        self.soil_min_threshold = soil_min_threshold
        self.green_max_threshold = green_max_threshold
    
    def detect_green_grass(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect green grass areas with expanded range
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (green area ratio, green mask)
        """
        # Convert to HSV space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define multiple green ranges to catch different types of grass
        masks = []
        
        # Range 1: Standard green grass (broader hue range)
        lower_green1 = np.array([25, 20, 30])    # Hue: 25-90 (includes yellow-green to blue-green)
        upper_green1 = np.array([90, 255, 255])  # Full saturation and brightness range
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        masks.append(mask1)
        
        # Range 2: Low saturation green (for overexposed/washed out grass)
        lower_green2 = np.array([25, 10, 100])   # Very low saturation, high brightness
        upper_green2 = np.array([90, 80, 255])   # Medium saturation, full brightness
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        masks.append(mask2)
        
        # Range 3: Dark/shadowed grass
        lower_green3 = np.array([25, 15, 15])    # Low brightness
        upper_green3 = np.array([90, 255, 80])   # Low to medium brightness
        mask3 = cv2.inRange(hsv, lower_green3, upper_green3)
        masks.append(mask3)
        
        # Combine all masks
        mask_green = masks[0]
        for mask in masks[1:]:
            mask_green = cv2.bitwise_or(mask_green, mask)
        
        # Lighter validation for green channel dominance
        b, g, r = cv2.split(image)
        # Green should be at least slightly stronger than red OR blue
        green_dominance_mask = ((g > r * 1.05) | (g > b * 1.05)) & (g > 30)
        
        # Apply green dominance check
        mask_green = cv2.bitwise_and(mask_green, green_dominance_mask.astype(np.uint8) * 255)
        
        # Morphological operations for noise removal
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        
        # Calculate green area ratio
        green_ratio = np.sum(mask_green > 0) / (image.shape[0] * image.shape[1])
        
        return float(green_ratio), mask_green
    
    def detect_soil_improved(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Improved soil detection for various soil types
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (soil area ratio, soil mask)
        """
        # Convert to HSV and LAB spaces for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Initialize empty mask
        total_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Method 1: Brown soil detection (expanded range)
        # Brown hues: 10-30, with varying saturation and value
        brown_masks = []
        # Light brown
        brown_masks.append(cv2.inRange(hsv, np.array([10, 10, 60]), np.array([30, 180, 200])))
        # Dark brown
        brown_masks.append(cv2.inRange(hsv, np.array([10, 20, 20]), np.array([25, 255, 100])))
        # Reddish brown
        brown_masks.append(cv2.inRange(hsv, np.array([0, 20, 20]), np.array([15, 180, 150])))
        
        for mask in brown_masks:
            total_mask = cv2.bitwise_or(total_mask, mask)
        
        # Method 2: Gray/neutral soil (low saturation)
        # For your image, this is crucial - many soils appear grayish
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 40, 180]))
        total_mask = cv2.bitwise_or(total_mask, gray_mask)
        
        # Method 3: Dark soil detection (low value)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 90]))
        total_mask = cv2.bitwise_or(total_mask, dark_mask)
        
        # Method 4: LAB color space detection
        # In LAB space, soil usually has:
        # - L (lightness): 20-70
        # - a (green-red): -10 to +20 (slightly towards red)
        # - b (blue-yellow): 0 to +30 (slightly towards yellow)
        lab_soil_mask = cv2.inRange(lab, np.array([20, 118, 128]), np.array([70, 138, 158]))
        total_mask = cv2.bitwise_or(total_mask, lab_soil_mask)
        
        # Method 5: Texture-based approach
        # Soil often has uniform texture, unlike grass which has more variation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local standard deviation
        kernel_size = 5
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        mean_sq = cv2.blur(gray**2, (kernel_size, kernel_size))
        std_dev = np.sqrt(mean_sq - mean**2)
        
        # Low texture variation areas (likely soil)
        low_texture_mask = (std_dev < 15).astype(np.uint8) * 255
        
        # Combine with color-based detection
        total_mask = cv2.bitwise_or(total_mask, low_texture_mask)
        
        # Remove green areas (definitely not soil)
        _, green_mask = self.detect_green_grass(image)
        total_mask = cv2.bitwise_and(total_mask, cv2.bitwise_not(green_mask))
        
        # Morphological operations to clean up
        kernel = np.ones((7, 7), np.uint8)
        total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_CLOSE, kernel)
        total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate soil area ratio
        soil_ratio = np.sum(total_mask > 0) / (image.shape[0] * image.shape[1])
        
        return float(soil_ratio), total_mask
    
    def detect_soil(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Original soil detection method (kept for compatibility)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (soil area ratio, soil mask)
        """
        # Use the improved method
        return self.detect_soil_improved(image)
    
    def is_in_work_area(self, image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """Determine if image is in work area
        
        Determination logic:
        1. If green ratio > 50%, considered grass (non-work area)
        2. If soil ratio > 30% and green ratio < 30%, considered work area
        3. Other cases determined by green and soil ratio combination
        
        Args:
            image: Input image
            
        Returns:
            (is in work area, detection metrics)
        """
        # Detect green grass
        green_ratio, green_mask = self.detect_green_grass(image)
        
        # Detect soil using improved method
        soil_ratio, soil_mask = self.detect_soil_improved(image)
        
        # Calculate other areas (may be sky, road, etc.)
        other_ratio = 1.0 - green_ratio - soil_ratio
        
        # Build metrics
        metrics = {
            'green_ratio': green_ratio,
            'soil_ratio': soil_ratio,
            'other_ratio': float(other_ratio),
            'green_soil_ratio': green_ratio / (soil_ratio + 0.001)  # Avoid division by zero
        }
        
        # Determination logic
        in_work_area = True
        reasons = []
        
        # Rule 1: Too much green, considered grass
        if green_ratio > self.grass_threshold:
            in_work_area = False
            reasons.append(f"too_much_green ({green_ratio:.1%})")
        
        # Rule 2: Sufficient soil and little green, considered work area
        elif soil_ratio > self.soil_min_threshold and green_ratio < self.green_max_threshold:
            in_work_area = True
            # This is typical work area
        
        # Rule 3: Too little soil (lowered threshold for better detection)
        elif soil_ratio < 0.05:  # Changed from 0.1 to 0.05
            in_work_area = False
            reasons.append(f"insufficient_soil ({soil_ratio:.1%})")
        
        # Rule 4: Both green and soil are minimal, may be other areas
        elif green_ratio < 0.1 and soil_ratio < 0.15:  # Changed from 0.2 to 0.15
            in_work_area = False
            reasons.append("not_farmland")
        
        metrics['in_work_area'] = in_work_area
        metrics['reasons'] = reasons
        
        return in_work_area, metrics
    
    def visualize_detection(self, image: np.ndarray) -> np.ndarray:
        """Visualize detection results
        
        Args:
            image: Input image
            
        Returns:
            Visualized result image
        """
        # Detect each area
        green_ratio, green_mask = self.detect_green_grass(image)
        soil_ratio, soil_mask = self.detect_soil_improved(image)
        
        # Create color mask
        result = image.copy()
        overlay = np.zeros_like(image)
        
        # Mark green areas as bright green
        overlay[green_mask > 0] = [0, 255, 0]
        
        # Mark soil areas as brown
        overlay[soil_mask > 0] = [42, 42, 165]  # Brown in BGR format
        
        # Blend original image and mask
        alpha = 0.3
        result = cv2.addWeighted(result, 1-alpha, overlay, alpha, 0)
        
        # Add text information
        in_area, metrics = self.is_in_work_area(image)
        status_text = "Work Area" if in_area else "Non-Work Area"
        color = (0, 255, 0) if in_area else (0, 0, 255)
        
        # Background box
        cv2.rectangle(result, (5, 5), (300, 110), (0, 0, 0), -1)
        cv2.rectangle(result, (5, 5), (300, 110), color, 2)
        
        # Text information
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
        """Get detailed color statistics
        
        Args:
            image: Input image
            
        Returns:
            Color statistics
        """
        in_area, metrics = self.is_in_work_area(image)
        
        # Add more statistical information
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Average hue, saturation, value
        metrics['mean_hue'] = float(np.mean(hsv[:, :, 0]))
        metrics['mean_saturation'] = float(np.mean(hsv[:, :, 1]))
        metrics['mean_value'] = float(np.mean(hsv[:, :, 2]))
        
        # Dominant hue analysis
        hue_hist, _ = np.histogram(hsv[:, :, 0], bins=18, range=(0, 180))
        dominant_hue_bin = np.argmax(hue_hist)
        metrics['dominant_hue'] = float(dominant_hue_bin * 10)  # Convert to degrees
        
        return metrics