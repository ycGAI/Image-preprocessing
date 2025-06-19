#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import logging
from typing import Dict, Optional

from sharpness_classifier import ImageSharpnessClassifier
from batch_processor import BatchImageProcessor
from report_generator import ReportGenerator
from file_utils import FileUtils


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"loading fail: {e}")
        return {}


def create_default_config() -> Dict:

    return {
        "source_root": "./input",
        "output_root": "./output",
        "classifier_params": {
            "laplacian_threshold": 100.0,
            "sobel_threshold": 50.0,
            "brenner_threshold": 1000.0,
            "tenengrad_threshold": 500.0,
            "variance_threshold": 50.0
        },
        "processing": {
            "max_workers": 4,
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        },
        "logging": {
            "level": "INFO",
            "file": None
        },
        "reports": {
            "generate_visual": True,
            "export_csv": True,
            "save_plots": True
        }
    }


def save_config_template(output_path: str):
    config = create_default_config()
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"config file saved {output_path}")
    except Exception as e:
        print(f"fail save {e}")


def main():
    parser = argparse.ArgumentParser(
        description="image sharpness classification tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example usage:
  python main.py --source ./images --output ./results
  python main.py --config config.json
  python main.py --preview --source ./images --max-folders 3
  python main.py --create-config config_template.json
        """
    )
    
    parser.add_argument('--source', '-s', type=str, help='source folder path')
    parser.add_argument('--output', '-o', type=str, help='output folder path')
    parser.add_argument('--config', '-c', type=str, help='config file path')
    parser.add_argument('--preview', action='store_true', help='preview mode (analyze only, do not process)')
    parser.add_argument('--max-folders', type=int, default=3, help='maximum number of folders in preview mode')
    parser.add_argument('--max-workers', type=int, help='maximum number of threads')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='log level')
    parser.add_argument('--log-file', type=str, help='log file path')
    parser.add_argument('--create-config', type=str, help='create config file template')
    parser.add_argument('--no-visual', action='store_true', help='do not generate visual report')
    parser.add_argument('--no-csv', action='store_true', help='do not export CSV file')
    
    args = parser.parse_args()

    if args.create_config:
        save_config_template(args.create_config)
        return

    if args.config:
        config = load_config(args.config)
        if not config:
            print("loading config failed, using default config")
            config = create_default_config()
    else:
        config = create_default_config()

    if args.source:
        config['source_root'] = args.source
    if args.output:
        config['output_root'] = args.output
    if args.max_workers:
        config['processing']['max_workers'] = args.max_workers
    if args.log_file:
        config['logging']['file'] = args.log_file
    if args.no_visual:
        config['reports']['generate_visual'] = False
    if args.no_csv:
        config['reports']['export_csv'] = False
    
    config['logging']['level'] = args.log_level

    setup_logging(config['logging']['level'], config['logging'].get('file'))
    logger = logging.getLogger(__name__)

    if not config.get('source_root'):
        logger.error("please specify the source folder path (--source or set in config file)")
        return 1
    
    if not config.get('output_root'):
        logger.error("please specify the output folder path (--output or set in config file)")
        return 1

    source_path = Path(config['source_root'])
    if not source_path.exists():
        logger.error(f"source folder does not exist: {source_path}")
        return 1
    
    # import ipdb; ipdb.set_trace()
    processor = BatchImageProcessor(
        source_root=config['source_root'],
        output_root=config['output_root'],
        classifier_params=config.get('classifier_params', {}),
        max_workers=config['processing']['max_workers'],
        supported_formats=config['processing']['supported_formats']
    )
    
    # 预览模式
    if args.preview:
        logger.info("preview mode.")
        preview_results = processor.preview_processing(args.max_folders)
        
        print("\n" + "="*50)
        print("Preview Results")
        print("="*50)
        print(f"Total found folders: {preview_results['total_found_folders']}")
        print(f"Preview folder count: {preview_results['preview_folder_count']}")
        print("-" * 50)
        
        for folder_info in preview_results['preview_folders']:
            print(f"Folder: {folder_info['folder_name']}")
            print(f"  Image-JSON pairs: {folder_info['image_json_txt_triples']}")
            print(f"  Path: {folder_info['folder_path']}")
            print()
        
        print("="*50)
        return 0

    logger.info("Starting batch processing...")
    try:
        results = processor.run()

        output_path = Path(config['output_root'])
        report_generator = ReportGenerator(output_path)
 
        processing_time = results.get('processing_time', 0)
        report = report_generator.generate_processing_report(
            results, 
            processing_time, 
            processor.classifier.get_thresholds()
        )

        detailed_analysis = report_generator.generate_detailed_analysis(results)
 
        if config['reports']['generate_visual']:
            try:
                visual_info = report_generator.generate_visual_report(
                    results, 
                    config['reports']['save_plots']
                )
                if visual_info:
                    logger.info("visual report generated successfully")
            except Exception as e:
                logger.warning(f"Error generating visual report: {e}")

        if config['reports']['export_csv']:
            try:
                csv_path = report_generator.export_to_csv(results)
                if csv_path:
                    logger.info(f"CSV file exported successfully: {csv_path}")
            except Exception as e:
                logger.warning(f"Error exporting CSV: {e}")

        report_generator.print_summary(report)
        
        logger.info("Batch processing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("User interrupted processing")
        return 1
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return 1


def run_interactive_mode():
    print("="*60)
    print("Image Sharpness Classifier - Interactive Mode")
    print("="*60)

    source_root = input("Please enter the source folder path: ").strip()
    if not source_root:
        print("Source folder path cannot be empty")
        return 1
    
    output_root = input("Please enter the output folder path: ").strip()
    if not output_root:
        print("Output folder path cannot be empty")  
        return 1

    use_default = input("Would you like to use default settings? (y/n, default y): ").strip().lower()
    
    config = create_default_config()
    config['source_root'] = source_root
    config['output_root'] = output_root
    
    if use_default not in ['y', 'yes', '']:
        try:
            max_workers = input(f"Maximum number of threads (default {config['processing']['max_workers']}): ").strip()
            if max_workers:
                config['processing']['max_workers'] = int(max_workers)

            print("\nClassifier threshold settings (press enter to use default):")
            thresholds = config['classifier_params']
            
            for key, default_value in thresholds.items():
                user_input = input(f"{key} (default {default_value}): ").strip()
                if user_input:
                    thresholds[key] = float(user_input)
                    
        except ValueError as e:
            print(f"Input value error: {e}")
            return 1

    setup_logging(config['logging']['level'])
    logger = logging.getLogger(__name__)
  
    source_path = Path(source_root)
    if not source_path.exists():
        print(f"Source folder does not exist:  {source_path}")
        return 1

    preview = input("Would you like to preview the processing results? (y/n, default n): ").strip().lower()

    processor = BatchImageProcessor(
        source_root=config['source_root'],
        output_root=config['output_root'],
        classifier_params=config['classifier_params'],
        max_workers=config['processing']['max_workers'],
        supported_formats=config['processing']['supported_formats']
    )
    
    if preview in ['y', 'yes']:
        print("\nPreviewing processing results...")
        preview_results = processor.preview_processing(3)
        
        print(f"\nFound {preview_results['total_found_folders']} time folders")
        for folder_info in preview_results['preview_folders']:
            print(f"- {folder_info['folder_name']}: {folder_info['image_json_txt_triples']} time-image pairs")
        
        continue_process = input("\nWould you like to continue with the full processing? (y/n): ").strip().lower()
        if continue_process not in ['y', 'yes']:
            print("Processing has been canceled")
            return 0

    print("\nStarting batch processing...")
    try:
        results = processor.run()
 
        output_path = Path(config['output_root'])
        report_generator = ReportGenerator(output_path)
        
        processing_time = results.get('processing_time', 0)
        report = report_generator.generate_processing_report(
            results, 
            processing_time, 
            processor.classifier.get_thresholds()
        )

        report_generator.print_summary(report)
        
        print(f"\nProcessing completed! Results saved at: {output_path}")
        return 0
        
    except KeyboardInterrupt:
        print("\nUser interrupted processing")
        return 1
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        exit_code = run_interactive_mode()
    else:
        exit_code = main()
    
    sys.exit(exit_code)