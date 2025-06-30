import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

from enhanced_image_processor import EnhancedImageProcessor
from enhanced_file_utils import FileOperationType
from enhanced_report_generator import EnhancedReportGenerator

logger = logging.getLogger(__name__)


class EnhancedBatchProcessor:
    """Enhanced batch image processor
    
    Supports sharpness, exposure, position and work area detection
    """
    
    def __init__(self,
                 source_root: str,
                 output_root: str,
                 classifier_params: Optional[Dict] = None,
                 exposure_params: Optional[Dict] = None,
                 position_params: Optional[Dict] = None,
                 work_area_params: Optional[Dict] = None,
                 file_operation: FileOperationType = 'copy',
                 max_workers: int = 4):
        """Initialize batch processor
        
        Args:
            source_root: Source folder root directory
            output_root: Output folder root directory
            classifier_params: Sharpness classifier parameters
            exposure_params: Exposure analyzer parameters
            position_params: Position detector parameters
            work_area_params: Work area detector parameters
            file_operation: File operation type
            max_workers: Maximum number of threads
        """
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.max_workers = max_workers
        self.file_operation = file_operation
        
        # Create processor
        self.processor = EnhancedImageProcessor(
            classifier_params=classifier_params,
            exposure_params=exposure_params,
            position_params=position_params,
            work_area_params=work_area_params,
            file_operation=file_operation
        )
        
        # Processing results
        self.results = {
            'processed_folders': [],
            'total_images': 0,
            'clean_images': 0,
            'dirty_images': 0,
            'same_position_groups': 0,
            'dirty_reasons_summary': {},
            'errors': []
        }
        
    def find_time_folders(self) -> List[Path]:
        """Find folders in time format"""
        time_folders = []
        
        for folder in self.source_root.iterdir():
            if folder.is_dir() and self.processor.file_utils.is_time_format(folder.name):
                time_folders.append(folder)
                
        return sorted(time_folders)
        
    def process_folder_wrapper(self, source_folder: Path) -> Dict:
        """Folder processing wrapper (for multithreading)
        
        Args:
            source_folder: Source folder
            
        Returns:
            Processing results
        """
        try:
            # Create output folder
            output_folder = self.output_root / source_folder.name
            
            # Process folder
            result = self.processor.process_folder(source_folder, output_folder)
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing folder {source_folder.name}: {str(e)}"
            logger.error(error_msg)
            return {
                'folder': source_folder.name,
                'error': str(e),
                'total_images': 0,
                'clean_images': 0,
                'dirty_images': 0
            }
            
    def run(self) -> Dict:
        """Run batch processing
        
        Returns:
            Processing results
        """
        start_time = time.time()
        logger.info("Starting enhanced batch image processing...")
        
        # Find time folders
        time_folders = self.find_time_folders()
        
        if not time_folders:
            logger.error("No time-formatted folders found")
            return self.results
            
        logger.info(f"Found {len(time_folders)} time folders")
        
        # Create output root directory
        self.processor.file_utils.create_directory(self.output_root)
        
        # Multi-threaded folder processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_folder = {
                executor.submit(self.process_folder_wrapper, folder): folder
                for folder in time_folders
            }
            
            # Collect results
            with tqdm(total=len(time_folders), desc="Processing folders") as pbar:
                for future in as_completed(future_to_folder):
                    folder = future_to_folder[future]
                    
                    try:
                        result = future.result()
                        self.results['processed_folders'].append(result)
                        
                        # Update total counts
                        self.results['total_images'] += result.get('total_images', 0)
                        self.results['clean_images'] += result.get('clean_images', 0)
                        self.results['dirty_images'] += result.get('dirty_images', 0)
                        self.results['same_position_groups'] += result.get('same_position_groups', 0)
                        
                        # Update dirty reason statistics
                        for reason, count in result.get('dirty_reasons_count', {}).items():
                            self.results['dirty_reasons_summary'][reason] = \
                                self.results['dirty_reasons_summary'].get(reason, 0) + count
                                
                        # Check for errors
                        if 'error' in result:
                            self.results['errors'].append(f"{folder.name}: {result['error']}")
                            
                    except Exception as e:
                        error_msg = f"Error collecting results {folder.name}: {str(e)}"
                        logger.error(error_msg)
                        self.results['errors'].append(error_msg)
                        
                    pbar.update(1)
                    
        # Calculate processing time
        self.results['processing_time'] = time.time() - start_time
        
        # Generate report
        self.generate_report()
        
        logger.info("Enhanced batch processing complete!")
        return self.results
        
    def generate_report(self):
        """Generate processing report"""
        report_generator = EnhancedReportGenerator(self.output_root)
        
        # Collect all settings
        settings = {
            'file_operation': self.file_operation,
            'sharpness_thresholds': self.processor.sharpness_classifier.get_thresholds(),
            'exposure_thresholds': {
                'overexposure_threshold': self.processor.exposure_analyzer.overexposure_threshold,
                'underexposure_threshold': self.processor.exposure_analyzer.underexposure_threshold
            },
            'position_distance_threshold': self.processor.position_detector.gps_distance_threshold,
            'position_rotation_threshold': self.processor.position_detector.rotation_threshold,
            'work_area_thresholds': {
                'grass_threshold': self.processor.work_area_detector.grass_threshold,
                'soil_min_threshold': self.processor.work_area_detector.soil_min_threshold,
                'green_max_threshold': self.processor.work_area_detector.green_max_threshold
            }
        }
        
        # Generate report
        report = report_generator.generate_processing_report(
            self.results, 
            self.results.get('processing_time', 0),
            settings
        )
        
        # Print summary
        report_generator.print_summary(report)
        
        # Generate visual report
        try:
            report_generator.generate_visual_report(self.results, save_plots=True)
            logger.info("Visual report generated successfully")
        except Exception as e:
            logger.warning(f"Error generating visual report: {e}")
            
        # Export CSV
        try:
            csv_path = report_generator.export_to_csv(self.results)
            if csv_path:
                logger.info(f"CSV file exported: {csv_path}")
        except Exception as e:
            logger.warning(f"Error exporting CSV: {e}")
            
    def preview_processing(self, max_folders: int = 3) -> Dict:
        """Preview processing results
        
        Args:
            max_folders: Maximum number of folders to preview
            
        Returns:
            Preview results
        """
        logger.info("Running preview mode...")
        
        # Find time folders
        time_folders = self.find_time_folders()
        
        if not time_folders:
            logger.error("No time-formatted folders found")
            return {
                'total_found_folders': 0,
                'preview_folder_count': 0,
                'preview_folders': []
            }
            
        # Select folders to preview
        preview_folders = time_folders[:max_folders]
        
        preview_results = {
            'total_found_folders': len(time_folders),
            'preview_folder_count': len(preview_folders),
            'preview_folders': []
        }
        
        for folder in preview_folders:
            # Find image-JSON pairs
            pairs = self.processor.file_utils.find_image_json_pairs(folder)
            
            folder_info = {
                'folder_name': folder.name,
                'folder_path': str(folder),
                'image_json_pairs': len(pairs)
            }
            
            # Analyze sample images (up to 5)
            if pairs:
                sample_pairs = pairs[:5]
                folder_info['sample_analysis'] = []
                
                for img_path, json_path in sample_pairs:
                    analysis = self.processor.quality_analyzer.analyze_image(img_path)
                    folder_info['sample_analysis'].append({
                        'image': img_path.name,
                        'is_clean': analysis['is_clean'],
                        'dirty_reasons': analysis.get('dirty_reasons', [])
                    })
                    
            preview_results['preview_folders'].append(folder_info)
            
        return preview_results