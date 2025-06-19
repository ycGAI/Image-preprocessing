import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

from sharpness_classifier import ImageSharpnessClassifier
from file_utils import FileUtils
from image_processor import ImageProcessor, ProcessingStats

logger = logging.getLogger(__name__)


class BatchImageProcessor:
    """批量图像处理器"""

    def __init__(self,
                 source_root: str,
                 output_root: str,
                 classifier_params: Optional[Dict] = None,
                 max_workers: int = 4,
                 supported_formats: Optional[List[str]] = None):
        """初始化批量处理器
        
        Args:
            source_root: 源文件夹根目录
            output_root: 输出文件夹根目录
            classifier_params: 分类器参数
            max_workers: 最大线程数
            supported_formats: 支持的文件格式
        """
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.max_workers = max_workers

        # 初始化组件
        if classifier_params is None:
            classifier_params = {}
        self.classifier = ImageSharpnessClassifier(**classifier_params)
        
        self.file_utils = FileUtils(supported_formats)
        self.image_processor = ImageProcessor(self.classifier, self.file_utils)
        self.stats = ProcessingStats()

        # 结果存储
        self.results = {
            'processed_folders': [],
            'total_images': 0,
            'sharp_images': 0,
            'blurry_images': 0,
            'errors': []
        }

    def process_single_image_wrapper(self, image_path: Path, json_path: Path, 
                                   output_folder: Path, txt_path: Optional[Path] = None) -> Dict:
        """单个图像处理包装器（用于线程池）
        
        Args:
            image_path: 图像文件路径
            json_path: JSON文件路径
            output_folder: 输出文件夹
            txt_path: TXT文件路径（可选）
            
        Returns:
            处理结果
        """
        # import ipdb; ipdb.set_trace()
        result = self.image_processor.process_single_image(
            image_path, json_path, output_folder, txt_path)
        
        # 更新统计信息
        self.stats.increment_processed()
        if result['status'] == 'success':
            if result['classification'] == "sharp":
                self.stats.increment_sharp()
            else:
                self.stats.increment_blurry()
        else:
            self.stats.increment_error()
            
        return result

    def process_folder(self, source_folder: Path) -> Dict:
        """处理单个时间文件夹
        
        Args:
            source_folder: 源文件夹路径
            
        Returns:
            文件夹处理结果
        """
        folder_name = source_folder.name
        logger.info(f"开始处理文件夹: {folder_name}")

        # 创建输出文件夹
        output_folder = self.output_root / folder_name
        self.file_utils.create_directory(output_folder)

        # 查找文件对
        # image_json_pairs = self.file_utils.find_image_json_pairs(source_folder)
        image_json_txt_triples = self.file_utils.find_image_json_txt_triples(source_folder)

        # 如果需要处理TXT文件，可以使用三元组
        # image_json_txt_triples = self.file_utils.find_image_json_txt_triples(source_folder)

        if not image_json_txt_triples:
            logger.warning(f"文件夹 {folder_name} 中没有找到图像-JSON对")
            return {
                'folder': folder_name,
                'processed': 0,
                'sharp': 0,
                'blurry': 0,
                'errors': 0
            }

        logger.info(f"文件夹 {folder_name} 中找到 {len(image_json_txt_triples)} 个图像-JSON对")

        # 多线程处理图像
        folder_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            # future_to_pair = {
            #     executor.submit(
            #         self.process_single_image_wrapper,
            #         img_path, json_path, output_folder
            #     ): (img_path, json_path)
            #     for img_path, json_path in image_json_pairs
            # }
            future_to_triple = {
                executor.submit(
                    self.process_single_image_wrapper,
                    img_path, json_path, output_folder, txt_path  # 添加txt_path参数
                ): (img_path, json_path, txt_path)
                for img_path, json_path, txt_path in image_json_txt_triples
            }

            # 收集结果
            with tqdm(total=len(image_json_txt_triples), desc=f"处理 {folder_name}") as pbar:
                for future in as_completed(future_to_triple):
                    result = future.result()
                    folder_results.append(result)
                    pbar.update(1)

        # 统计文件夹结果
        folder_stats = {
            'folder': folder_name,
            'processed': len([r for r in folder_results if r['status'] == 'success']),
            'sharp': len([r for r in folder_results if r.get('classification') == 'sharp']),
            'blurry': len([r for r in folder_results if r.get('classification') == 'blurry']),
            'errors': len([r for r in folder_results if r['status'] == 'error'])
        }

        logger.info(f"文件夹 {folder_name} 处理完成: {folder_stats}")
        return folder_stats

    def run(self) -> Dict:
        """运行批量处理
        
        Returns:
            处理结果
        """
        start_time = time.time()
        logger.info("开始批量图像清晰度分类处理...")

        # 重置统计信息
        self.stats.reset()

        # 查找所有时间文件夹
        time_folders = self.file_utils.find_time_folders(self.source_root)

        if not time_folders:
            logger.error("没有找到时间格式的文件夹")
            return self.results

        # 创建输出根目录
        self.file_utils.create_directory(self.output_root)

        # 处理每个文件夹
        for folder in time_folders:
            try:
                folder_result = self.process_folder(folder)
                self.results['processed_folders'].append(folder_result)

                # 更新总计数
                self.results['total_images'] += folder_result['processed']
                self.results['sharp_images'] += folder_result['sharp']
                self.results['blurry_images'] += folder_result['blurry']

            except Exception as e:
                error_msg = f"处理文件夹 {folder.name} 时出错: {str(e)}"
                logger.error(error_msg)
                self.results['errors'].append(error_msg)

        # 计算总耗时
        total_time = time.time() - start_time

        self.results['processing_time'] = total_time

        # 生成处理报告
        self.generate_report(total_time)

        logger.info("批量处理完成！")
        return self.results

    def run_with_custom_filter(self, folder_filter_func=None) -> Dict:
        """使用自定义过滤器运行批量处理
        
        Args:
            folder_filter_func: 文件夹过滤函数，接收Path对象，返回bool
            
        Returns:
            处理结果
        """
        start_time = time.time()
        logger.info("开始自定义批量图像清晰度分类处理...")

        # 重置统计信息
        self.stats.reset()

        # 查找所有符合条件的文件夹
        if folder_filter_func is None:
            folders = self.file_utils.find_time_folders(self.source_root)
        else:
            all_folders = [f for f in self.source_root.iterdir() if f.is_dir()]
            folders = [f for f in all_folders if folder_filter_func(f)]

        if not folders:
            logger.error("没有找到符合条件的文件夹")
            return self.results

        # 创建输出根目录
        self.file_utils.create_directory(self.output_root)

        # 处理每个文件夹
        for folder in folders:
            try:
                folder_result = self.process_folder(folder)
                self.results['processed_folders'].append(folder_result)

                # 更新总计数
                self.results['total_images'] += folder_result['processed']
                self.results['sharp_images'] += folder_result['sharp']
                self.results['blurry_images'] += folder_result['blurry']

            except Exception as e:
                error_msg = f"处理文件夹 {folder.name} 时出错: {str(e)}"
                logger.error(error_msg)
                self.results['errors'].append(error_msg)

        # 计算总耗时
        total_time = time.time() - start_time

        # 生成处理报告
        self.generate_report(total_time)

        logger.info("自定义批量处理完成！")
        return self.results

    def generate_report(self, total_time: float):
        """生成处理报告
        
        Args:
            total_time: 总处理时间
        """
        report = {
            'processing_summary': {
                'total_time_seconds': round(total_time, 2),
                'total_folders': len(self.results['processed_folders']),
                'total_images': self.results['total_images'],
                'sharp_images': self.results['sharp_images'],
                'blurry_images': self.results['blurry_images'],
                'error_count': len(self.results['errors']),
                'processing_speed': round(self.results['total_images'] / total_time, 2) if total_time > 0 else 0,
                'sharp_ratio': round(self.results['sharp_images'] / self.results['total_images'], 3) if self.results['total_images'] > 0 else 0
            },
            'folder_details': self.results['processed_folders'],
            'classifier_settings': self.classifier.get_thresholds(),
            'errors': self.results['errors'],
            'processing_info': {
                'max_workers': self.max_workers,
                'supported_formats': self.file_utils.supported_formats
            }
        }

        # 保存报告
        report_path = self.output_root / 'processing_report.json'
        self.file_utils.write_json_file(report_path, report)
        logger.info(f"处理报告已保存到: {report_path}")

        # 打印摘要
        self._print_summary(report)

    def _print_summary(self, report: Dict):
        """打印处理摘要
        
        Args:
            report: 报告数据
        """
        summary = report['processing_summary']
        print("\n" + "="*50)
        print("批量处理摘要")
        print("="*50)
        print(f"总耗时: {summary['total_time_seconds']} 秒")
        print(f"处理文件夹数: {summary['total_folders']}")
        print(f"总图像数: {summary['total_images']}")
        print(f"清晰图像: {summary['sharp_images']} ({summary['sharp_ratio']:.1%})")
        print(f"模糊图像: {summary['blurry_images']}")
        print(f"错误数: {summary['error_count']}")
        print(f"处理速度: {summary['processing_speed']} 图像/秒")
        print("="*50)

    def get_current_stats(self) -> Dict:
        """获取当前统计信息
        
        Returns:
            当前统计信息
        """
        return self.stats.get_stats()

    def update_classifier_thresholds(self, **kwargs):
        """更新分类器阈值
        
        Args:
            **kwargs: 阈值参数
        """
        self.classifier.update_thresholds(**kwargs)
        logger.info(f"分类器阈值已更新: {kwargs}")

    def set_max_workers(self, max_workers: int):
        """设置最大线程数
        
        Args:
            max_workers: 最大线程数
        """
        self.max_workers = max_workers
        logger.info(f"最大线程数设置为: {max_workers}")

    def preview_processing(self, max_folders: int = 3) -> Dict:
        """预览处理结果（只处理前几个文件夹）
        
        Args:
            max_folders: 最大处理文件夹数
            
        Returns:
            预览结果
        """
        logger.info(f"开始预览处理（最多处理 {max_folders} 个文件夹）...")
        
        time_folders = self.file_utils.find_time_folders(self.source_root)
        preview_folders = time_folders[:max_folders]
        
        preview_results = {
            'preview_folders': [],
            'total_found_folders': len(time_folders),
            'preview_folder_count': len(preview_folders)
        }
        
        for folder in preview_folders:
            pairs = self.file_utils.image_json_txt_triples(folder)
            preview_results['preview_folders'].append({
                'folder_name': folder.name,
                'image_json_txt_triples': len(pairs),
                'folder_path': str(folder)
            })
        
        return preview_results