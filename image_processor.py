import cv2
import json
import time
from pathlib import Path
from typing import Dict, Optional
from threading import Lock
import logging

from sharpness_classifier import ImageSharpnessClassifier
from file_utils import FileUtils

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, classifier: ImageSharpnessClassifier, file_utils: FileUtils):
        self.classifier = classifier
        self.file_utils = file_utils
        
    def process_single_image(self, image_path: Path, json_path: Path, 
                           output_folder: Path, txt_path: Optional[Path] = None) -> Dict:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"unable to read image: {image_path}")

            classification, metrics = self.classifier.classify_with_ensemble(image)

            json_data = self.file_utils.read_json_file(json_path)
            # json_data['sharpness_analysis'] = {
            #     'classification': classification,
            #     'metrics': {k: float(v) for k, v in metrics.items()},
            #     'thresholds': self.classifier.get_thresholds(),
            #     'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S')
            # }
            target_category = "sharp" if classification == "sharp" else "blurry"
            target_folder = output_folder / target_category
            self.file_utils.create_directory(target_folder)

            target_image_path = target_folder / image_path.name
            self.file_utils.copy_file_safely(image_path, target_image_path)


            target_json_path = target_folder / json_path.name
            self.file_utils.write_json_file(target_json_path, json_data)

            
            if txt_path and txt_path.exists():
                target_txt_path = target_folder / txt_path.name
                self.file_utils.copy_file_safely(txt_path, target_txt_path)

            return {
                'status': 'success',
                'image_path': str(image_path),
                'classification': classification,
                'metrics': metrics,
                'output_folder': str(target_folder)
            }

        except Exception as e:
            error_msg = f"processing failed {image_path}: {str(e)}"
            logger.error(error_msg)

            return {
                'status': 'error',
                'image_path': str(image_path),
                'error': str(e)
            }

    def process_image_with_single_method(self, image_path: Path, method: str) -> Dict:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"unable to read image: {image_path}")

            classification, metric_value = self.classifier.classify_single_method(image, method)

            return {
                'status': 'success',
                'image_path': str(image_path),
                'method': method,
                'classification': classification,
                'metric_value': metric_value,
                'threshold': self.classifier.thresholds[method]
            }

        except Exception as e:
            error_msg = f"processing failed {image_path}: {str(e)}"
            logger.error(error_msg)

            return {
                'status': 'error',
                'image_path': str(image_path),
                'method': method,
                'error': str(e)
            }

    def batch_analyze_images(self, image_paths: list, method: str = 'ensemble') -> Dict:
        results = []
        sharp_count = 0
        error_count = 0

        for image_path in image_paths:
            if method == 'ensemble':
                try:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError(f"unable to read image: {image_path}")
                    
                    classification, metrics = self.classifier.classify_with_ensemble(image)
                    result = {
                        'status': 'success',
                        'image_path': str(image_path),
                        'classification': classification,
                        'metrics': metrics
                    }
                    if classification == "sharp":
                        sharp_count += 1
                        
                except Exception as e:
                    result = {
                        'status': 'error',
                        'image_path': str(image_path),
                        'error': str(e)
                    }
                    error_count += 1
            else:
                result = self.process_image_with_single_method(Path(image_path), method)
                if result['status'] == 'success' and result['classification'] == "sharp":
                    sharp_count += 1
                elif result['status'] == 'error':
                    error_count += 1

            results.append(result)

        return {
            'total_images': len(image_paths),
            'sharp_images': sharp_count,
            'blurry_images': len(image_paths) - sharp_count - error_count,
            'error_count': error_count,
            'method': method,
            'results': results
        }


class ProcessingStats:
    def __init__(self):
        self.lock = Lock()
        self.processed_count = 0
        self.sharp_count = 0
        self.blurry_count = 0
        self.error_count = 0

    def increment_processed(self):
        with self.lock:
            self.processed_count += 1

    def increment_sharp(self):
        with self.lock:
            self.sharp_count += 1

    def increment_blurry(self):
        with self.lock:
            self.blurry_count += 1

    def increment_error(self):
        with self.lock:
            self.error_count += 1

    def get_stats(self) -> Dict:
        with self.lock:
            return {
                'processed': self.processed_count,
                'sharp': self.sharp_count,
                'blurry': self.blurry_count,
                'errors': self.error_count
            }

    def reset(self):
        with self.lock:
            self.processed_count = 0
            self.sharp_count = 0
            self.blurry_count = 0
            self.error_count = 0