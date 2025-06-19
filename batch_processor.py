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

    def __init__(self,
                 source_root: str,
                 output_root: str,
                 classifier_params: Optional[Dict] = None,
                 max_workers: int = 4,
                 supported_formats: Optional[List[str]] = None):

        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.max_workers = max_workers

        if classifier_params is None:
            classifier_params = {}
        self.classifier = ImageSharpnessClassifier(**classifier_params)
        
        self.file_utils = FileUtils(supported_formats)
        self.image_processor = ImageProcessor(self.classifier, self.file_utils)
        self.stats = ProcessingStats()

        self.results = {
            'processed_folders': [],
            'total_images': 0,
            'sharp_images': 0,
            'blurry_images': 0,
            'errors': []
        }

    def process_single_image_wrapper(self, image_path: Path, json_path: Path, 
                                   output_folder: Path, txt_path: Optional[Path] = None) -> Dict:
        # import ipdb; ipdb.set_trace()
        result = self.image_processor.process_single_image(
            image_path, json_path, output_folder, txt_path)
        
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
        folder_name = source_folder.name
        logger.info(f"start processing folder: {folder_name}")

        output_folder = self.output_root / folder_name
        self.file_utils.create_directory(output_folder)

        # image_json_pairs = self.file_utils.find_image_json_pairs(source_folder)
        image_json_txt_triples = self.file_utils.find_image_json_txt_triples(source_folder)

        # image_json_txt_triples = self.file_utils.find_image_json_txt_triples(source_folder)

        if not image_json_txt_triples:
            logger.warning(f"file {folder_name} not found or no image-JSON pairs")
            return {
                'folder': folder_name,
                'processed': 0,
                'sharp': 0,
                'blurry': 0,
                'errors': 0
            }

        logger.info(f"文件夹 {folder_name} 中找到 {len(image_json_txt_triples)} 个图像-JSON对")

        folder_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

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
                    img_path, json_path, output_folder, txt_path  
                ): (img_path, json_path, txt_path)
                for img_path, json_path, txt_path in image_json_txt_triples
            }

            with tqdm(total=len(image_json_txt_triples), desc=f"处理 {folder_name}") as pbar:
                for future in as_completed(future_to_triple):
                    result = future.result()
                    folder_results.append(result)
                    pbar.update(1)

        folder_stats = {
            'folder': folder_name,
            'processed': len([r for r in folder_results if r['status'] == 'success']),
            'sharp': len([r for r in folder_results if r.get('classification') == 'sharp']),
            'blurry': len([r for r in folder_results if r.get('classification') == 'blurry']),
            'errors': len([r for r in folder_results if r['status'] == 'error'])
        }

        logger.info(f"file {folder_name} processed: {folder_stats}")
        return folder_stats

    def run(self) -> Dict:
        start_time = time.time()
        logger.info("start batch image sharpness classification processing...")

        self.stats.reset()

        time_folders = self.file_utils.find_time_folders(self.source_root)

        if not time_folders:
            logger.error("not found any time folders")
            return self.results

        self.file_utils.create_directory(self.output_root)

        for folder in time_folders:
            try:
                folder_result = self.process_folder(folder)
                self.results['processed_folders'].append(folder_result)

                self.results['total_images'] += folder_result['processed']
                self.results['sharp_images'] += folder_result['sharp']
                self.results['blurry_images'] += folder_result['blurry']

            except Exception as e:
                error_msg = f"process folder {folder.name} encountered an error: {str(e)}"
                logger.error(error_msg)
                self.results['errors'].append(error_msg)

        total_time = time.time() - start_time

        self.results['processing_time'] = total_time

        self.generate_report(total_time)

        logger.info("batch processing completed!")
        return self.results

    def run_with_custom_filter(self, folder_filter_func=None) -> Dict:
        start_time = time.time()
        logger.info("start custom batch image sharpness classification processing...")

        self.stats.reset()

        if folder_filter_func is None:
            folders = self.file_utils.find_time_folders(self.source_root)
        else:
            all_folders = [f for f in self.source_root.iterdir() if f.is_dir()]
            folders = [f for f in all_folders if folder_filter_func(f)]

        if not folders:
            logger.error("not found any folders matching the filter")
            return self.results

        self.file_utils.create_directory(self.output_root)

        for folder in folders:
            try:
                folder_result = self.process_folder(folder)
                self.results['processed_folders'].append(folder_result)

                self.results['total_images'] += folder_result['processed']
                self.results['sharp_images'] += folder_result['sharp']
                self.results['blurry_images'] += folder_result['blurry']

            except Exception as e:
                error_msg = f"process folder  {folder.name} encountered an error: {str(e)}"
                logger.error(error_msg)
                self.results['errors'].append(error_msg)
        total_time = time.time() - start_time

        self.generate_report(total_time)

        logger.info("custom batch processing completed!")
        return self.results

    def generate_report(self, total_time: float):

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

        report_path = self.output_root / 'processing_report.json'
        self.file_utils.write_json_file(report_path, report)
        logger.info(f"Report saved to:  {report_path}")

        self._print_summary(report)

    def _print_summary(self, report: Dict):
  
        summary = report['processing_summary']
        print("\n" + "="*50)
        print("Batch Processing Summary")
        print("="*50)
        print(f"Total processing time: {summary['total_time_seconds']} seconds")
        print(f"Processed folder count: {summary['total_folders']}")
        print(f"Total images: {summary['total_images']}")
        print(f"Sharp images: {summary['sharp_images']} ({summary['sharp_ratio']:.1%})")
        print(f"Blurry images: {summary['blurry_images']}")
        print(f"Error count: {summary['error_count']}")
        print(f"Processing speed: {summary['processing_speed']} images/second")
        print("="*50)

    def get_current_stats(self) -> Dict:

        return self.stats.get_stats()

    def update_classifier_thresholds(self, **kwargs):
        self.classifier.update_thresholds(**kwargs)
        logger.info(f"thresholds updated: {kwargs}")

    def set_max_workers(self, max_workers: int):
        self.max_workers = max_workers
        logger.info(f"max workers set to: {max_workers}")

    def preview_processing(self, max_folders: int = 3) -> Dict:

        logger.info(f"starting preview processing (up to {max_folders} folders)...")
        
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