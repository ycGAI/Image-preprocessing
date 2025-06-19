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
    """单个图像处理器"""

    def __init__(self, classifier: ImageSharpnessClassifier, file_utils: FileUtils):
        """初始化图像处理器
        
        Args:
            classifier: 清晰度分类器
            file_utils: 文件工具
        """
        self.classifier = classifier
        self.file_utils = file_utils
        
    def process_single_image(self, image_path: Path, json_path: Path, 
                           output_folder: Path, txt_path: Optional[Path] = None) -> Dict:
        """处理单个图像
        
        Args:
            image_path: 图像文件路径
            json_path: JSON文件路径
            output_folder: 输出文件夹
            txt_path: TXT文件路径（可选）
            
        Returns:
            处理结果字典
        """
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

            # 进行清晰度分类
            classification, metrics = self.classifier.classify_with_ensemble(image)

            # 读取JSON文件
            json_data = self.file_utils.read_json_file(json_path)

            # 添加清晰度分析结果
            json_data['sharpness_analysis'] = {
                'classification': classification,
                'metrics': {k: float(v) for k, v in metrics.items()},
                'thresholds': self.classifier.get_thresholds(),
                'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # 确定目标文件夹
            target_category = "清晰" if classification == "清晰" else "模糊"
            target_folder = output_folder / target_category
            self.file_utils.create_directory(target_folder)

            # 复制图像文件
            target_image_path = target_folder / image_path.name
            self.file_utils.copy_file_safely(image_path, target_image_path)

            # 保存更新后的JSON文件
            target_json_path = target_folder / json_path.name
            self.file_utils.write_json_file(target_json_path, json_data)

            # 如果有TXT文件，也复制过去
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
            error_msg = f"处理图像失败 {image_path}: {str(e)}"
            logger.error(error_msg)

            return {
                'status': 'error',
                'image_path': str(image_path),
                'error': str(e)
            }

    def process_image_with_single_method(self, image_path: Path, method: str) -> Dict:
        """使用单一方法处理图像
        
        Args:
            image_path: 图像文件路径
            method: 分类方法
            
        Returns:
            处理结果字典
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

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
            error_msg = f"处理图像失败 {image_path}: {str(e)}"
            logger.error(error_msg)

            return {
                'status': 'error',
                'image_path': str(image_path),
                'method': method,
                'error': str(e)
            }

    def batch_analyze_images(self, image_paths: list, method: str = 'ensemble') -> Dict:
        """批量分析图像
        
        Args:
            image_paths: 图像路径列表
            method: 分析方法（'ensemble' 或具体方法名）
            
        Returns:
            批量分析结果
        """
        results = []
        sharp_count = 0
        error_count = 0

        for image_path in image_paths:
            if method == 'ensemble':
                try:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError(f"无法读取图像: {image_path}")
                    
                    classification, metrics = self.classifier.classify_with_ensemble(image)
                    result = {
                        'status': 'success',
                        'image_path': str(image_path),
                        'classification': classification,
                        'metrics': metrics
                    }
                    if classification == "清晰":
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
                if result['status'] == 'success' and result['classification'] == "清晰":
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
    """处理统计信息"""

    def __init__(self):
        self.lock = Lock()
        self.processed_count = 0
        self.sharp_count = 0
        self.blurry_count = 0
        self.error_count = 0

    def increment_processed(self):
        """增加已处理计数"""
        with self.lock:
            self.processed_count += 1

    def increment_sharp(self):
        """增加清晰图像计数"""
        with self.lock:
            self.sharp_count += 1

    def increment_blurry(self):
        """增加模糊图像计数"""
        with self.lock:
            self.blurry_count += 1

    def increment_error(self):
        """增加错误计数"""
        with self.lock:
            self.error_count += 1

    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            return {
                'processed': self.processed_count,
                'sharp': self.sharp_count,
                'blurry': self.blurry_count,
                'errors': self.error_count
            }

    def reset(self):
        """重置统计信息"""
        with self.lock:
            self.processed_count = 0
            self.sharp_count = 0
            self.blurry_count = 0
            self.error_count = 0