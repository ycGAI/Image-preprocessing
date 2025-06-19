import cv2
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageSharpnessClassifier:

    def __init__(self,
                 laplacian_threshold: float = 100.0,
                 sobel_threshold: float = 50.0,
                 brenner_threshold: float = 1000.0,
                 tenengrad_threshold: float = 500.0,
                 variance_threshold: float = 50.0):

        self.thresholds = {
            'laplacian': laplacian_threshold,
            'sobel': sobel_threshold,
            'brenner': brenner_threshold,
            'tenengrad': tenengrad_threshold,
            'variance': variance_threshold
        }

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def laplacian_variance(self, image: np.ndarray) -> float:
        gray = self._convert_to_gray(image)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def sobel_variance(self, image: np.ndarray) -> float:
        gray = self._convert_to_gray(image)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        return sobel.var()

    def brenner_gradient(self, image: np.ndarray) -> float:
        gray = self._convert_to_gray(image)
        diff = np.diff(gray.astype(np.float64), axis=1)
        return np.sum(diff**2)

    def tenengrad_variance(self, image: np.ndarray) -> float:
        gray = self._convert_to_gray(image)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        threshold = np.mean(gradient_magnitude)
        return np.sum(gradient_magnitude[gradient_magnitude > threshold]**2)

    def calculate_all_metrics(self, image: np.ndarray) -> Dict[str, float]:
        metrics = {}
        metrics['laplacian'] = self.laplacian_variance(image)
        metrics['sobel'] = self.sobel_variance(image)
        metrics['brenner'] = self.brenner_gradient(image)
        metrics['tenengrad'] = self.tenengrad_variance(image)
        return metrics

    def classify_with_ensemble(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        metrics = self.calculate_all_metrics(image)
        votes = []
        
        for method in ['laplacian', 'sobel', 'brenner', 'tenengrad']:
            if method in metrics:
                is_sharp = metrics[method] > self.thresholds[method]
                votes.append(1 if is_sharp else 0)
        
        sharp_votes = sum(votes)
        classification = "sharp" if sharp_votes > len(votes) / 2 else "blurry"
        
        return classification, metrics

    def classify_single_method(self, image: np.ndarray, method: str) -> Tuple[str, float]:
        if method not in self.thresholds:
            raise ValueError(f"Unsupported method: {method}")
        
        if method == 'laplacian':
            metric_value = self.laplacian_variance(image)
        elif method == 'sobel':
            metric_value = self.sobel_variance(image)
        elif method == 'brenner':
            metric_value = self.brenner_gradient(image)
        elif method == 'tenengrad':
            metric_value = self.tenengrad_variance(image)
        
        classification = "sharp" if metric_value > self.thresholds[method] else "blurry"
        return classification, metric_value

    def update_thresholds(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.thresholds:
                self.thresholds[key] = value
            else:
                logger.warning(f"Unknown threshold parameter: {key}")

    def get_thresholds(self) -> Dict[str, float]:
        return self.thresholds.copy()