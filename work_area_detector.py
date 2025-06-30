import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class WorkAreaDetector:
    """Work area detector - Simplified version
    
    Determination by color:
    - Work area: Mostly soil (brown/dark), with minimal plants
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
        """Detect green grass areas
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (green area ratio, green mask)
        """
        # Convert to HSV space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define grass green range (relatively broad)
        # Grass green is usually quite bright
        lower_green = np.array([25, 30, 30])    # Hue 25-85 covers various greens
        upper_green = np.array([85, 255, 255])
        
        # Create green mask
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations for noise removal
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        
        # Calculate green area ratio
        green_ratio = np.sum(mask_green > 0) / (image.shape[0] * image.shape[1])
        
        return float(green_ratio), mask_green
    
    def detect_soil(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect soil areas (brown and dark areas)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (soil area ratio, soil mask)
        """
        # Convert to HSV space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define multiple soil color ranges
        masks = []
        
        # 1. Brown soil (hue 10-25)
        lower_brown = np.array([10, 20, 20])
        upper_brown = np.array([25, 255, 180])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        masks.append(mask_brown)
        
        # 2. Dark soil (low brightness)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])  # Brightness < 80
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        masks.append(mask_dark)
        
        # 3. Gray soil (low saturation)
        lower_gray = np.array([0, 0, 40])
        upper_gray = np.array([180, 30, 120])  # Saturation < 30
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        masks.append(mask_gray)
        
        # Merge all soil masks
        mask_soil = masks[0]
        for mask in masks[1:]:
            mask_soil = cv2.bitwise_or(mask_soil, mask)
        
        # Remove overlapping green parts
        _, mask_green = self.detect_green_grass(image)
        mask_soil = cv2.bitwise_and(mask_soil, cv2.bitwise_not(mask_green))
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask_soil = cv2.morphologyEx(mask_soil, cv2.MORPH_OPEN, kernel)
        mask_soil = cv2.morphologyEx(mask_soil, cv2.MORPH_CLOSE, kernel)
        
        # Calculate soil area ratio
        soil_ratio = np.sum(mask_soil > 0) / (image.shape[0] * image.shape[1])
        
        return float(soil_ratio), mask_soil
    
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
        
        # Detect soil
        soil_ratio, soil_mask = self.detect_soil(image)
        
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
        
        # Rule 3: Too little soil
        elif soil_ratio < 0.1:
            in_work_area = False
            reasons.append(f"insufficient_soil ({soil_ratio:.1%})")
        
        # Rule 4: Both green and soil are minimal, may be other areas (sky, buildings, etc.)
        elif green_ratio < 0.1 and soil_ratio < 0.2:
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
        soil_ratio, soil_mask = self.detect_soil(image)
        
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