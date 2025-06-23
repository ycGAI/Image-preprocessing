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
        result = self.image_processor.process_single_image(
            image_path, json_path, output_folder, txt_path)
        
        # 更新统计信息
        self.stats.increment_processed()
        if result['status'] == 'success':
            if result['classification'] == "清晰":
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

        # 直接在这里实现文件查找逻辑，支持图像+JSON(+可选TXT)
        files = list(source_folder.glob('*'))
        
        # 按文件名分组
        file_groups = {}
        for file in files:
            if file.is_file():
                stem = file.stem
                if stem not in file_groups:
                    file_groups[stem] = []
                file_groups[stem].append(file)

        # 查找图像-JSON对，可能包含TXT
        image_json_pairs = []
        for stem, file_list in file_groups.items():
            image_file = None
            json_file = None
            txt_file = None

            for file in file_list:
                ext = file.suffix.lower()
                if ext in self.file_utils.supported_formats:
                    image_file = file
                elif ext == '.json':
                    json_file = file
                elif ext == '.txt':
                    txt_file = file

            # 只要有图像和JSON就添加（TXT是可选的）
            if image_file and json_file:
                image_json_pairs.append((image_file, json_file, txt_file))

        if not image_json_pairs:
            logger.warning(f"文件夹 {folder_name} 中没有找到图像-JSON对")
            return {
                'folder': folder_name,
                'processed': 0,
                'sharp': 0,
                'blurry': 0,
                'errors': 0
            }

        # 统计文件类型
        with_txt = sum(1 for _, _, txt in image_json_pairs if txt is not None)
        without_txt = len(image_json_pairs) - with_txt
        
        logger.info(f"文件夹 {folder_name} 中找到 {len(image_json_pairs)} 个图像-JSON对")
        logger.info(f"  - 包含TXT文件: {with_txt} 个")
        logger.info(f"  - 仅有JSON文件: {without_txt} 个")

        # 多线程处理图像
        folder_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_pair = {}
            for img_path, json_path, txt_path in image_json_pairs:
                future = executor.submit(
                    self.process_single_image_wrapper,
                    img_path, json_path, output_folder, txt_path
                )
                future_to_pair[future] = (img_path, json_path, txt_path)

            # 收集结果
            with tqdm(total=len(image_json_pairs), desc=f"处理 {folder_name}") as pbar:
                for future in as_completed(future_to_pair):
                    result = future.result()
                    folder_results.append(result)
                    pbar.update(1)

        # 统计文件夹结果
        folder_stats = {
            'folder': folder_name,
            'processed': len([r for r in folder_results if r['status'] == 'success']),
            'sharp': len([r for r in folder_results if r.get('classification') == '清晰']),
            'blurry': len([r for r in folder_results if r.get('classification') == '模糊']),
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
            pairs = self.file_utils.find_image_json_pairs(folder)
            triples = self.file_utils.find_image_json_txt_triples(folder)
            
            # 计算实际需要处理的文件数（避免重复计算）
            processed_images = set()
            total_to_process = 0
            
            # 先统计三元组中的图像
            for img_path, json_path, txt_path in triples:
                processed_images.add(img_path)
                total_to_process += 1
            
            # 再统计只有图像-JSON对的情况
            for img_path, json_path in pairs:
                if img_path not in processed_images:
                    total_to_process += 1
            
            preview_results['preview_folders'].append({
                'folder_name': folder.name,
                'image_json_pairs': len(pairs),
                'image_json_txt_triples': len(triples),
                'total_to_process': total_to_process,
                'folder_path': str(folder)
            })
        
        return preview_results