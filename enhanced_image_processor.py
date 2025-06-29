import cv2
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from sharpness_classifier import ImageSharpnessClassifier
from exposure_analyzer import ExposureAnalyzer
from position_detector import PositionDetector
from work_area_detector import WorkAreaDetector
from enhanced_file_utils import EnhancedFileUtils, FileOperationType

logger = logging.getLogger(__name__)


class ImageQualityAnalyzer:
    """综合图像质量分析器"""
    
    def __init__(self,
                 sharpness_classifier: ImageSharpnessClassifier,
                 exposure_analyzer: ExposureAnalyzer,
                 work_area_detector: WorkAreaDetector):
        """初始化质量分析器
        
        Args:
            sharpness_classifier: 清晰度分类器
            exposure_analyzer: 曝光分析器
            work_area_detector: 工作区域检测器
        """
        self.sharpness_classifier = sharpness_classifier
        self.exposure_analyzer = exposure_analyzer
        self.work_area_detector = work_area_detector
        
    def analyze_image(self, image_path: Path) -> Dict:
        """分析单张图像的质量
        
        Args:
            image_path: 图像路径
            
        Returns:
            分析结果字典
        """
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
                
            result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'is_clean': True,
                'dirty_reasons': [],
                'metrics': {}
            }
            
            # 1. 清晰度分析
            sharpness_class, sharpness_metrics = self.sharpness_classifier.classify_with_ensemble(image)
            result['metrics']['sharpness'] = sharpness_metrics
            result['sharpness_classification'] = sharpness_class
            
            if sharpness_class == "模糊":
                result['is_clean'] = False
                result['dirty_reasons'].append('blurry')
                
            # 2. 曝光分析
            exposure_status, exposure_metrics = self.exposure_analyzer.analyze_exposure(image)
            result['metrics']['exposure'] = exposure_metrics
            result['exposure_status'] = exposure_status
            
            if exposure_status == 'overexposed':
                result['is_clean'] = False
                result['dirty_reasons'].append('overexposed')
            elif exposure_status == 'underexposed':
                result['is_clean'] = False
                result['dirty_reasons'].append('underexposed')
                
            # 3. 工作区域检测
            in_work_area, work_area_metrics = self.work_area_detector.is_in_work_area(image)
            result['metrics']['work_area'] = work_area_metrics
            result['in_work_area'] = in_work_area
            
            if not in_work_area:
                result['is_clean'] = False
                result['dirty_reasons'].append('out_of_work_area')
                
            return result
            
        except Exception as e:
            logger.error(f"分析图像失败 {image_path}: {str(e)}")
            return {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'is_clean': False,
                'dirty_reasons': ['analysis_error'],
                'error': str(e)
            }


class EnhancedImageProcessor:
    """增强的图像处理器
    
    整合清晰度、曝光、位置和工作区域检测
    """
    
    def __init__(self,
                 classifier_params: Optional[Dict] = None,
                 exposure_params: Optional[Dict] = None,
                 position_params: Optional[Dict] = None,
                 work_area_params: Optional[Dict] = None,
                 file_operation: FileOperationType = 'copy'):
        """初始化增强处理器
        
        Args:
            classifier_params: 清晰度分类器参数
            exposure_params: 曝光分析器参数
            position_params: 位置检测器参数
            work_area_params: 工作区域检测器参数
            file_operation: 文件操作类型
        """
        # 初始化各个组件
        self.sharpness_classifier = ImageSharpnessClassifier(
            **(classifier_params or {})
        )
        self.exposure_analyzer = ExposureAnalyzer(
            **(exposure_params or {})
        )
        self.position_detector = PositionDetector(
            **(position_params or {})
        )
        self.work_area_detector = WorkAreaDetector(
            **(work_area_params or {})
        )
        
        # 文件工具
        self.file_utils = EnhancedFileUtils(operation_type=file_operation)
        
        # 质量分析器
        self.quality_analyzer = ImageQualityAnalyzer(
            self.sharpness_classifier,
            self.exposure_analyzer,
            self.work_area_detector
        )
        
    def process_folder(self, source_folder: Path, output_folder: Path) -> Dict:
        """处理单个文件夹
        
        Args:
            source_folder: 源文件夹
            output_folder: 输出文件夹
            
        Returns:
            处理结果
        """
        logger.info(f"开始处理文件夹: {source_folder.name}")
        
        # 创建输出目录结构
        clean_folder = output_folder / "clean_data"
        dirty_folder = output_folder / "dirty_data"
        self.file_utils.create_directory(clean_folder)
        self.file_utils.create_directory(dirty_folder)
        
        # 查找图像-JSON对
        image_json_pairs = self.file_utils.find_image_json_pairs(source_folder)
        
        if not image_json_pairs:
            logger.warning(f"文件夹 {source_folder.name} 中没有找到图像-JSON对")
            return {
                'folder': source_folder.name,
                'total_images': 0,
                'clean_images': 0,
                'dirty_images': 0,
                'same_position_groups': 0
            }
            
        # 分析所有图像
        logger.info(f"分析 {len(image_json_pairs)} 张图像...")
        analysis_results = []
        image_paths = []
        
        for img_path, json_path in image_json_pairs:
            # 检查是否有对应的txt文件
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                txt_path = None
                
            analysis = self.quality_analyzer.analyze_image(img_path)
            analysis['json_path'] = json_path
            analysis['txt_path'] = txt_path
            analysis_results.append(analysis)
            image_paths.append(img_path)
            
        # 检测同位置拍摄的图像组
        logger.info("检测同位置拍摄的图像...")
        # 提取图像路径和JSON路径
        image_paths = [pair[0] for pair in image_json_pairs]
        json_paths = [pair[1] for pair in image_json_pairs]
        
        same_position_groups = self.position_detector.detect_same_position_groups(
            image_paths, json_paths
        )
        
        # 处理同位置图像组
        same_position_images = set()
        for group in same_position_groups:
            # 保留前两张到clean_data，其余标记为dirty
            for i, img_path in enumerate(group):
                same_position_images.add(str(img_path))
                if i >= 2:  # 第三张及以后的图片
                    # 在分析结果中标记为dirty
                    for result in analysis_results:
                        if result['image_path'] == str(img_path):
                            result['is_clean'] = False
                            if 'same_position_extra' not in result['dirty_reasons']:
                                result['dirty_reasons'].append('same_position_extra')
                            break
                            
        # 统计并移动文件
        stats = {
            'folder': source_folder.name,
            'total_images': len(analysis_results),
            'clean_images': 0,
            'dirty_images': 0,
            'same_position_groups': len(same_position_groups),
            'dirty_reasons_count': {}
        }
        
        for result in analysis_results:
            img_path = Path(result['image_path'])
            json_path = result['json_path']
            txt_path = result.get('txt_path')
            
            # 更新JSON文件
            json_data = self.file_utils.read_json_file(json_path)
            json_data['quality_analysis'] = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'is_clean': result['is_clean'],
                'dirty_reasons': result['dirty_reasons'],
                'sharpness_classification': result.get('sharpness_classification'),
                'exposure_status': result.get('exposure_status'),
                'in_work_area': result.get('in_work_area'),
                'metrics': result['metrics']
            }
            
            # 确定目标文件夹
            if result['is_clean']:
                target_folder = clean_folder
                stats['clean_images'] += 1
            else:
                target_folder = dirty_folder
                stats['dirty_images'] += 1
                # 统计dirty原因
                for reason in result['dirty_reasons']:
                    stats['dirty_reasons_count'][reason] = stats['dirty_reasons_count'].get(reason, 0) + 1
                    
            # 传输文件
            self.file_utils.transfer_file_pair(
                img_path, json_path, target_folder, txt_path
            )
            
            # 保存更新后的JSON
            target_json_path = target_folder / json_path.name
            self.file_utils.write_json_file(target_json_path, json_data)
            
        logger.info(f"文件夹 {source_folder.name} 处理完成: "
                   f"总计{stats['total_images']}张, "
                   f"干净{stats['clean_images']}张, "
                   f"脏数据{stats['dirty_images']}张")
        
        return stats
    
    def generate_analysis_report(self, analysis_results: List[Dict]) -> Dict:
        """生成分析报告
        
        Args:
            analysis_results: 分析结果列表
            
        Returns:
            报告字典
        """
        report = {
            'total_analyzed': len(analysis_results),
            'quality_distribution': {
                'clean': 0,
                'dirty': 0
            },
            'dirty_reasons_distribution': {},
            'exposure_distribution': {
                'normal': 0,
                'overexposed': 0,
                'underexposed': 0
            },
            'sharpness_distribution': {
                'sharp': 0,
                'blurry': 0
            },
            'work_area_distribution': {
                'in_area': 0,
                'out_of_area': 0
            }
        }
        
        for result in analysis_results:
            # 质量分布
            if result['is_clean']:
                report['quality_distribution']['clean'] += 1
            else:
                report['quality_distribution']['dirty'] += 1
                
            # Dirty原因分布
            for reason in result.get('dirty_reasons', []):
                report['dirty_reasons_distribution'][reason] = \
                    report['dirty_reasons_distribution'].get(reason, 0) + 1
                    
            # 曝光分布
            exposure = result.get('exposure_status', 'normal')
            report['exposure_distribution'][exposure] += 1
            
            # 清晰度分布
            sharpness = result.get('sharpness_classification', '清晰')
            if sharpness == '清晰':
                report['sharpness_distribution']['sharp'] += 1
            else:
                report['sharpness_distribution']['blurry'] += 1
                
            # 工作区域分布
            if result.get('in_work_area', True):
                report['work_area_distribution']['in_area'] += 1
            else:
                report['work_area_distribution']['out_of_area'] += 1
                
        return report