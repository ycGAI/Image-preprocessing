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
    """增强的批量图像处理器
    
    支持清晰度、曝光、位置和工作区域检测
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
        """初始化批量处理器
        
        Args:
            source_root: 源文件夹根目录
            output_root: 输出文件夹根目录
            classifier_params: 清晰度分类器参数
            exposure_params: 曝光分析器参数
            position_params: 位置检测器参数
            work_area_params: 工作区域检测器参数
            file_operation: 文件操作类型
            max_workers: 最大线程数
        """
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.max_workers = max_workers
        self.file_operation = file_operation
        
        # 创建处理器
        self.processor = EnhancedImageProcessor(
            classifier_params=classifier_params,
            exposure_params=exposure_params,
            position_params=position_params,
            work_area_params=work_area_params,
            file_operation=file_operation
        )
        
        # 处理结果
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
        """查找时间格式的文件夹"""
        time_folders = []
        
        for folder in self.source_root.iterdir():
            if folder.is_dir() and self.processor.file_utils.is_time_format(folder.name):
                time_folders.append(folder)
                
        return sorted(time_folders)
        
    def process_folder_wrapper(self, source_folder: Path) -> Dict:
        """文件夹处理包装器（用于多线程）
        
        Args:
            source_folder: 源文件夹
            
        Returns:
            处理结果
        """
        try:
            # 创建输出文件夹
            output_folder = self.output_root / source_folder.name
            
            # 处理文件夹
            result = self.processor.process_folder(source_folder, output_folder)
            
            return result
            
        except Exception as e:
            error_msg = f"处理文件夹 {source_folder.name} 时出错: {str(e)}"
            logger.error(error_msg)
            return {
                'folder': source_folder.name,
                'error': str(e),
                'total_images': 0,
                'clean_images': 0,
                'dirty_images': 0
            }
            
    def run(self) -> Dict:
        """运行批量处理
        
        Returns:
            处理结果
        """
        start_time = time.time()
        logger.info("开始增强批量图像处理...")
        
        # 查找时间文件夹
        time_folders = self.find_time_folders()
        
        if not time_folders:
            logger.error("没有找到时间格式的文件夹")
            return self.results
            
        logger.info(f"找到 {len(time_folders)} 个时间文件夹")
        
        # 创建输出根目录
        self.processor.file_utils.create_directory(self.output_root)
        
        # 多线程处理文件夹
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_folder = {
                executor.submit(self.process_folder_wrapper, folder): folder
                for folder in time_folders
            }
            
            # 收集结果
            with tqdm(total=len(time_folders), desc="处理文件夹") as pbar:
                for future in as_completed(future_to_folder):
                    folder = future_to_folder[future]
                    
                    try:
                        result = future.result()
                        self.results['processed_folders'].append(result)
                        
                        # 更新总计数
                        self.results['total_images'] += result.get('total_images', 0)
                        self.results['clean_images'] += result.get('clean_images', 0)
                        self.results['dirty_images'] += result.get('dirty_images', 0)
                        self.results['same_position_groups'] += result.get('same_position_groups', 0)
                        
                        # 更新dirty原因统计
                        for reason, count in result.get('dirty_reasons_count', {}).items():
                            self.results['dirty_reasons_summary'][reason] = \
                                self.results['dirty_reasons_summary'].get(reason, 0) + count
                                
                        # 检查错误
                        if 'error' in result:
                            self.results['errors'].append(f"{folder.name}: {result['error']}")
                            
                    except Exception as e:
                        error_msg = f"收集结果时出错 {folder.name}: {str(e)}"
                        logger.error(error_msg)
                        self.results['errors'].append(error_msg)
                        
                    pbar.update(1)
                    
        # 计算处理时间
        self.results['processing_time'] = time.time() - start_time
        
        # 生成报告
        self.generate_report()
        
        logger.info("增强批量处理完成！")
        return self.results
        
    def generate_report(self):
        """生成处理报告"""
        report_generator = EnhancedReportGenerator(self.output_root)
        
        # 收集所有设置
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
        
        # 生成报告
        report = report_generator.generate_processing_report(
            self.results, 
            self.results.get('processing_time', 0),
            settings
        )
        
        # 打印摘要
        report_generator.print_summary(report)
        
        # 生成可视化报告
        try:
            report_generator.generate_visual_report(self.results, save_plots=True)
            logger.info("可视化报告生成成功")
        except Exception as e:
            logger.warning(f"生成可视化报告时出错: {e}")
            
        # 导出CSV
        try:
            csv_path = report_generator.export_to_csv(self.results)
            if csv_path:
                logger.info(f"CSV文件已导出: {csv_path}")
        except Exception as e:
            logger.warning(f"导出CSV时出错: {e}")
            
        return preview_results