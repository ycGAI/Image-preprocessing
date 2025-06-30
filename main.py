#!/usr/bin/env python3
"""
Enhanced Image Quality Classification Main Program

New features:
- Exposure detection (overexposure/underexposure)
- Same position shot detection
- Work area detection (plants and soil)
- Supports file operation modes (copy/move/symlink)

Usage:
    python enhanced_main.py --source /path/to/source --output /path/to/output
    python enhanced_main.py --config enhanced_config.json
    python enhanced_main.py --preview --source /path/to/source
"""

import argparse
import json
import sys
from pathlib import Path
import logging
from typing import Dict, Optional

from enhanced_batch_processor import EnhancedBatchProcessor
from enhanced_file_utils import FileOperationType

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
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
    """Load configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        return {}


def create_default_config() -> Dict:
    """Create default configuration"""
    return {
        "source_root": "./input",
        "output_root": "./output",
        "file_operation": "copy",  # copy, move, symlink
        "classifier_params": {
            "laplacian_threshold": 100.0,
            "sobel_threshold": 50.0,
            "brenner_threshold": 1000.0,
            "tenengrad_threshold": 500.0,
            "variance_threshold": 50.0
        },
        "exposure_params": {
            "overexposure_threshold": 0.05,
            "underexposure_threshold": 0.05,
            "bright_pixel_threshold": 240,
            "dark_pixel_threshold": 15
        },
        "position_params": {
            "gps_distance_threshold": 2.0,
            "rotation_threshold": 0.1
        },
        "work_area_params": {
            "grass_threshold": 0.5,          # Modified: Grass detection threshold
            "soil_min_threshold": 0.3,      # Modified: Minimum soil ratio
            "green_max_threshold": 0.3       # Modified: Maximum green ratio for work area
        },
        "processing": {
            "max_workers": 4
        },
        "logging": {
            "level": "INFO",
            "file": None
        }
    }


def save_config_template(output_path: str):
    """Save configuration file template"""
    config = create_default_config()
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Enhanced config file template saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save config file template: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Enhanced Image Quality Classification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Feature description:
  - Detect blurry images
  - Detect overexposed and underexposed images
  - Detect images taken consecutively at the same position
  - Detect images outside work area (non-plant/soil)
  - Support file copy, move or symlink

Usage examples:
  python enhanced_main.py --source ./images --output ./results
  python enhanced_main.py --config enhanced_config.json --file-operation move
  python enhanced_main.py --preview --source ./images --max-folders 3
  python enhanced_main.py --create-config enhanced_config_template.json
        """
    )
    
    # Basic parameters
    parser.add_argument('--source', '-s', type=str, help='Source folder path')
    parser.add_argument('--output', '-o', type=str, help='Output folder path')
    parser.add_argument('--config', '-c', type=str, help='Configuration file path')
    
    # File operation parameters
    parser.add_argument('--file-operation', choices=['copy', 'move', 'symlink'],
                       default='copy', help='File operation type')
    
    # Preview and template parameters
    parser.add_argument('--preview', action='store_true', help='Preview mode (analyze only, no processing)')
    parser.add_argument('--max-folders', type=int, default=3, help='Maximum folders in preview mode')
    parser.add_argument('--create-config', type=str, help='Create configuration file template')
    
    # Processing parameters
    parser.add_argument('--max-workers', type=int, help='Maximum number of threads')
    
    # Logging parameters
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    # Threshold adjustment parameters (optional)
    parser.add_argument('--sharpness-threshold', type=float, help='Sharpness threshold')
    parser.add_argument('--overexposure-threshold', type=float, help='Overexposure threshold')
    parser.add_argument('--underexposure-threshold', type=float, help='Underexposure threshold')
    parser.add_argument('--gps-distance-threshold', type=float, help='GPS distance threshold (meters)')
    parser.add_argument('--rotation-threshold', type=float, help='Rotation difference threshold')
    parser.add_argument('--grass-threshold', type=float, help='Grass detection threshold (0-1)')
    parser.add_argument('--soil-threshold', type=float, help='Minimum soil ratio threshold (0-1)')
    parser.add_argument('--work-area-green-max', type=float, help='Maximum green ratio for work area (0-1)')

    
    args = parser.parse_args()
    
    # Create configuration file template
    if args.create_config:
        save_config_template(args.create_config)
        return 0
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        if not config:
            print("Failed to load config file, using default config")
            config = create_default_config()
    else:
        config = create_default_config()
    
    # Command line parameters override config file
    if args.source:
        config['source_root'] = args.source
    if args.output:
        config['output_root'] = args.output
    if args.file_operation:
        config['file_operation'] = args.file_operation
    if args.max_workers:
        config['processing']['max_workers'] = args.max_workers
    if args.log_file:
        config['logging']['file'] = args.log_file
        
    # Threshold parameter overrides
    if args.sharpness_threshold:
        config['classifier_params']['laplacian_threshold'] = args.sharpness_threshold
    if args.overexposure_threshold:
        config['exposure_params']['overexposure_threshold'] = args.overexposure_threshold
    if args.underexposure_threshold:
        config['exposure_params']['underexposure_threshold'] = args.underexposure_threshold
    if args.gps_distance_threshold:
        config['position_params']['gps_distance_threshold'] = args.gps_distance_threshold
    if args.rotation_threshold:
        config['position_params']['rotation_threshold'] = args.rotation_threshold
    if args.grass_threshold:
        config['work_area_params']['grass_threshold'] = args.grass_threshold
    if args.soil_threshold:
        config['work_area_params']['soil_min_threshold'] = args.soil_threshold
    if args.work_area_green_max:
        config['work_area_params']['green_max_threshold'] = args.work_area_green_max
    
    config['logging']['level'] = args.log_level
    
    # Setup logging
    setup_logging(config['logging']['level'], config['logging'].get('file'))
    logger = logging.getLogger(__name__)
    
    # Validate required parameters
    if not config.get('source_root'):
        logger.error("Please specify source folder path (--source or set in config file)")
        return 1
    
    if not config.get('output_root'):
        logger.error("Please specify output folder path (--output or set in config file)")
        return 1
    
    # Check if source folder exists
    source_path = Path(config['source_root'])
    if not source_path.exists():
        logger.error(f"Source folder does not exist: {source_path}")
        return 1
    
    # Create batch processor
    processor = EnhancedBatchProcessor(
        source_root=config['source_root'],
        output_root=config['output_root'],
        classifier_params=config.get('classifier_params', {}),
        exposure_params=config.get('exposure_params', {}),
        position_params=config.get('position_params', {}),
        work_area_params=config.get('work_area_params', {}),
        file_operation=config.get('file_operation', 'copy'),
        max_workers=config['processing']['max_workers']
    )
    
    # Preview mode
    if args.preview:
        logger.info("Running preview mode...")
        preview_results = processor.preview_processing(args.max_folders)
        
        print("\n" + "="*60)
        print("Preview Results")
        print("="*60)
        print(f"Total folders found: {preview_results['total_found_folders']}")
        print(f"Preview folder count: {preview_results['preview_folder_count']}")
        print("-" * 60)
        
        for folder_info in preview_results['preview_folders']:
            print(f"\nFolder: {folder_info['folder_name']}")
            print(f"  Image-JSON pairs: {folder_info['image_json_pairs']}")
            print(f"  Path: {folder_info['folder_path']}")
            
            if 'sample_analysis' in folder_info:
                print("  Sample analysis:")
                for sample in folder_info['sample_analysis']:
                    status = "clean" if sample['is_clean'] else f"dirty ({', '.join(sample['dirty_reasons'])})"
                    print(f"    - {sample['image']}: {status}")
        
        print("\n" + "="*60)
        return 0
    
    # Formal processing
    logger.info("Starting enhanced batch processing...")
    logger.info(f"File operation mode: {config['file_operation']}")
    
    try:
        results = processor.run()
        
        logger.info("Enhanced batch processing complete!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("User interrupted processing")
        return 1
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


def run_interactive_mode():
    """Interactive running mode"""
    print("="*60)
    print("Enhanced Image Quality Classification Tool - Interactive Mode")
    print("="*60)
    
    # Get user input
    source_root = input("Please enter source folder path: ").strip()
    if not source_root:
        print("Source folder path cannot be empty")
        return 1
    
    output_root = input("Please enter output folder path: ").strip()
    if not output_root:
        print("Output folder path cannot be empty")  
        return 1
    
    # File operation type
    print("\nFile operation type:")
    print("1. Copy - default")
    print("2. Move")
    print("3. Symlink")
    operation_choice = input("Please select (1-3, default 1): ").strip()
    
    operation_map = {'1': 'copy', '2': 'move', '3': 'symlink', '': 'copy'}
    file_operation = operation_map.get(operation_choice, 'copy')
    
    # Ask whether to use default settings
    use_default = input("\nUse default detection parameters? (y/n, default y): ").strip().lower()
    
    config = create_default_config()
    config['source_root'] = source_root
    config['output_root'] = output_root
    config['file_operation'] = file_operation
    
    if use_default not in ['y', 'yes', '']:
        # Custom settings
        print("\nCustom detection parameters (press Enter for default):")
        
        try:
            # Sharpness threshold
            sharp_threshold = input(f"Sharpness threshold (default {config['classifier_params']['laplacian_threshold']}): ").strip()
            if sharp_threshold:
                config['classifier_params']['laplacian_threshold'] = float(sharp_threshold)
            
            # Exposure thresholds
            over_threshold = input(f"Overexposure threshold (default {config['exposure_params']['overexposure_threshold']}): ").strip()
            if over_threshold:
                config['exposure_params']['overexposure_threshold'] = float(over_threshold)
                
            under_threshold = input(f"Underexposure threshold (default {config['exposure_params']['underexposure_threshold']}): ").strip()
            if under_threshold:
                config['exposure_params']['underexposure_threshold'] = float(under_threshold)
            
            # Position detection thresholds
            gps_threshold = input(f"GPS distance threshold, meters (default {config['position_params']['gps_distance_threshold']}): ").strip()
            if gps_threshold:
                config['position_params']['gps_distance_threshold'] = float(gps_threshold)
                
            rotation_threshold = input(f"Rotation difference threshold (default {config['position_params']['rotation_threshold']}): ").strip()
            if rotation_threshold:
                config['position_params']['rotation_threshold'] = float(rotation_threshold)
            
            # Work area thresholds
            green_threshold = input(f"Vegetation detection threshold (default {config['work_area_params']['green_threshold']}): ").strip()
            if green_threshold:
                config['work_area_params']['green_threshold'] = float(green_threshold)
                
            # Thread count
            max_workers = input(f"Maximum thread count (default {config['processing']['max_workers']}): ").strip()
            if max_workers:
                config['processing']['max_workers'] = int(max_workers)
                
        except ValueError as e:
            print(f"Input value error: {e}")
            return 1
    
    # Setup logging
    setup_logging(config['logging']['level'])
    logger = logging.getLogger(__name__)
    
    # Check source folder
    source_path = Path(source_root)
    if not source_path.exists():
        print(f"Source folder does not exist: {source_path}")
        return 1
    
    # Ask whether to preview first
    preview = input("\nPreview processing results first? (y/n, default n): ").strip().lower()
    
    # Create processor
    processor = EnhancedBatchProcessor(
        source_root=config['source_root'],
        output_root=config['output_root'],
        classifier_params=config['classifier_params'],
        exposure_params=config['exposure_params'],
        position_params=config['position_params'],
        work_area_params=config['work_area_params'],
        file_operation=config['file_operation'],
        max_workers=config['processing']['max_workers']
    )
    
    if preview in ['y', 'yes']:
        print("\nPreviewing processing results...")
        preview_results = processor.preview_processing(3)
        
        print(f"\nFound {preview_results['total_found_folders']} time folders")
        for folder_info in preview_results['preview_folders'][:3]:
            print(f"\nFolder: {folder_info['folder_name']}")
            print(f"  Image pairs: {folder_info['image_json_pairs']}")
            
            if 'sample_analysis' in folder_info:
                print("  Sample analysis:")
                for sample in folder_info['sample_analysis']:
                    status = "clean" if sample['is_clean'] else f"dirty"
                    if not sample['is_clean']:
                        reasons = {
                            'blurry': 'Blurry',
                            'overexposed': 'Overexposed',
                            'underexposed': 'Underexposed',
                            'out_of_work_area': 'Out of work area',
                            'same_position_extra': 'Same position duplicate'
                        }
                        reason_list = [reasons.get(r, r) for r in sample['dirty_reasons']]
                        status += f" ({', '.join(reason_list)})"
                    print(f"    - {sample['image']}: {status}")
        
        continue_process = input("\nContinue with full processing? (y/n): ").strip().lower()
        if continue_process not in ['y', 'yes']:
            print("Processing cancelled")
            return 0
    
    # Start processing
    print(f"\nStarting batch processing (file operation: {file_operation})...")
    try:
        results = processor.run()
        
        print(f"\nProcessing complete! Results saved in: {output_root}")
        return 0
        
    except KeyboardInterrupt:
        print("\nUser interrupted processing")
        return 1
    except Exception as e:
        print(f"\nError during processing: {e}")
        return 1


if __name__ == "__main__":
    # If no command line arguments, enter interactive mode
    if len(sys.argv) == 1:
        exit_code = run_interactive_mode()
    else:
        exit_code = main()
    
    sys.exit(exit_code)