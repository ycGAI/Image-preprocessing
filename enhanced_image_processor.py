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
    """Comprehensive image quality analyzer"""
    
    def __init__(self,
                 sharpness_classifier: ImageSharpnessClassifier,
                 exposure_analyzer: ExposureAnalyzer,
                 work_area_detector: WorkAreaDetector):
        """Initialize quality analyzer
        
        Args:
            sharpness_classifier: Sharpness classifier
            exposure_analyzer: Exposure analyzer
            work_area_detector: Work area detector
        """
        self.sharpness_classifier = sharpness_classifier
        self.exposure_analyzer = exposure_analyzer
        self.work_area_detector = work_area_detector
        
    def analyze_image(self, image_path: Path) -> Dict:
        """Analyze quality of a single image
        
        Args:
            image_path: Image path
            
        Returns:
            Analysis result dictionary
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
                
            result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'is_clean': True,
                'dirty_reasons': [],
                'metrics': {}
            }
            
            # 1. Sharpness analysis
            sharpness_class, sharpness_metrics = self.sharpness_classifier.classify_with_ensemble(image)
            result['metrics']['sharpness'] = sharpness_metrics
            result['sharpness_classification'] = sharpness_class
            
            if sharpness_class == "blurry":
                result['is_clean'] = False
                result['dirty_reasons'].append('blurry')
                
            # 2. Exposure analysis
            exposure_status, exposure_metrics = self.exposure_analyzer.analyze_exposure(image)
            result['metrics']['exposure'] = exposure_metrics
            result['exposure_status'] = exposure_status
            
            if exposure_status == 'overexposed':
                result['is_clean'] = False
                result['dirty_reasons'].append('overexposed')
            elif exposure_status == 'underexposed':
                result['is_clean'] = False
                result['dirty_reasons'].append('underexposed')
                
            # 3. Work area detection
            in_work_area, work_area_metrics = self.work_area_detector.is_in_work_area(image)
            result['metrics']['work_area'] = work_area_metrics
            result['in_work_area'] = in_work_area
            
            if not in_work_area:
                result['is_clean'] = False
                result['dirty_reasons'].append('out_of_work_area')
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {str(e)}")
            return {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'is_clean': False,
                'dirty_reasons': ['analysis_error'],
                'error': str(e)
            }


class EnhancedImageProcessor:
    """Enhanced image processor
    
    Integrates sharpness, exposure, position and work area detection
    """
    
    def __init__(self,
                 classifier_params: Optional[Dict] = None,
                 exposure_params: Optional[Dict] = None,
                 position_params: Optional[Dict] = None,
                 work_area_params: Optional[Dict] = None,
                 file_operation: FileOperationType = 'copy'):
        """Initialize enhanced processor
        
        Args:
            classifier_params: Sharpness classifier parameters
            exposure_params: Exposure analyzer parameters
            position_params: Position detector parameters
            work_area_params: Work area detector parameters
            file_operation: File operation type
        """
        # Initialize components
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
        
        # File utilities
        self.file_utils = EnhancedFileUtils(operation_type=file_operation)
        
        # Quality analyzer
        self.quality_analyzer = ImageQualityAnalyzer(
            self.sharpness_classifier,
            self.exposure_analyzer,
            self.work_area_detector
        )
        
    def process_folder(self, source_folder: Path, output_folder: Path) -> Dict:
        """Process a single folder
        
        Args:
            source_folder: Source folder
            output_folder: Output folder
            
        Returns:
            Processing results
        """
        logger.info(f"Starting to process folder: {source_folder.name}")
        
        # Create output directory structure
        clean_folder = output_folder / "clean_data"
        dirty_folder = output_folder / "dirty_data"
        self.file_utils.create_directory(clean_folder)
        self.file_utils.create_directory(dirty_folder)
        
        # Find image-JSON pairs
        image_json_pairs = self.file_utils.find_image_json_pairs(source_folder)
        
        if not image_json_pairs:
            logger.warning(f"No image-JSON pairs found in folder {source_folder.name}")
            return {
                'folder': source_folder.name,
                'total_images': 0,
                'clean_images': 0,
                'dirty_images': 0,
                'same_position_groups': 0
            }
            
        # Analyze all images
        logger.info(f"Analyzing {len(image_json_pairs)} images...")
        analysis_results = []
        image_paths = []
        
        for img_path, json_path in image_json_pairs:
            # Check if there is a corresponding txt file
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                txt_path = None
                
            analysis = self.quality_analyzer.analyze_image(img_path)
            analysis['json_path'] = json_path
            analysis['txt_path'] = txt_path
            analysis_results.append(analysis)
            image_paths.append(img_path)
            
        # Detect same position image groups
        logger.info("Detecting same position images...")
        # Extract image paths and JSON paths
        image_paths = [pair[0] for pair in image_json_pairs]
        json_paths = [pair[1] for pair in image_json_pairs]
        
        same_position_groups = self.position_detector.detect_same_position_groups(
            image_paths, json_paths
        )
        
        # Process same position image groups
        same_position_images = set()
        for group in same_position_groups:
            # Keep first two to clean_data, mark the rest as dirty
            for i, img_path in enumerate(group):
                same_position_images.add(str(img_path))
                if i >= 2:  # Third image and onwards
                    # Mark as dirty in analysis results
                    for result in analysis_results:
                        if result['image_path'] == str(img_path):
                            result['is_clean'] = False
                            if 'same_position_extra' not in result['dirty_reasons']:
                                result['dirty_reasons'].append('same_position_extra')
                            break
                            
        # Statistics and file transfer
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
            
            # Update JSON file
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
            
            # Determine target folder
            if result['is_clean']:
                target_folder = clean_folder
                stats['clean_images'] += 1
            else:
                target_folder = dirty_folder
                stats['dirty_images'] += 1
                # Count dirty reasons
                for reason in result['dirty_reasons']:
                    stats['dirty_reasons_count'][reason] = stats['dirty_reasons_count'].get(reason, 0) + 1
                    
            # Transfer files
            self.file_utils.transfer_file_pair(
                img_path, json_path, target_folder, txt_path
            )
            
            # Save updated JSON
            target_json_path = target_folder / json_path.name
            self.file_utils.write_json_file(target_json_path, json_data)
            
        logger.info(f"Folder {source_folder.name} processing complete: "
                   f"Total {stats['total_images']} images, "
                   f"Clean {stats['clean_images']} images, "
                   f"Dirty {stats['dirty_images']} images")
        
        return stats
    
    def generate_analysis_report(self, analysis_results: List[Dict]) -> Dict:
        """Generate analysis report
        
        Args:
            analysis_results: List of analysis results
            
        Returns:
            Report dictionary
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
            # Quality distribution
            if result['is_clean']:
                report['quality_distribution']['clean'] += 1
            else:
                report['quality_distribution']['dirty'] += 1
                
            # Dirty reason distribution
            for reason in result.get('dirty_reasons', []):
                report['dirty_reasons_distribution'][reason] = \
                    report['dirty_reasons_distribution'].get(reason, 0) + 1
                    
            # Exposure distribution
            exposure = result.get('exposure_status', 'normal')
            report['exposure_distribution'][exposure] += 1
            
            # Sharpness distribution
            sharpness = result.get('sharpness_classification', 'sharp')
            if sharpness == 'sharp':
                report['sharpness_distribution']['sharp'] += 1
            else:
                report['sharpness_distribution']['blurry'] += 1
                
            # Work area distribution
            if result.get('in_work_area', True):
                report['work_area_distribution']['in_area'] += 1
            else:
                report['work_area_distribution']['out_of_area'] += 1
                
        return report