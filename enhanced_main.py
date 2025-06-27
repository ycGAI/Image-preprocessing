#!/usr/bin/env python3
"""
增强的图像质量分类主程序

新增功能:
- 曝光检测（过曝/欠曝）
- 同位置拍摄检测
- 工作区域检测（植物和土壤）
- 支持文件操作模式（复制/移动/软链接）

使用方法:
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
    """设置日志配置"""
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
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return {}


def create_default_config() -> Dict:
    """创建默认配置"""
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
            "similarity_threshold": 0.95,
            "feature_method": "orb",  # orb, sift, template
            "max_features": 500,
            "histogram_weight": 0.3,
            "structural_weight": 0.7
        },
        "work_area_params": {
            "green_threshold": 0.15,
            "brown_threshold": 0.10,
            "vegetation_index_threshold": 0.1,
            "texture_threshold": 20.0,
            "min_valid_area": 0.3
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
    """保存配置文件模板"""
    config = create_default_config()
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"增强配置文件模板已保存到: {output_path}")
    except Exception as e:
        print(f"保存配置文件模板失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="增强的图像质量分类工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明:
  - 检测模糊图像
  - 检测过曝和欠曝光图像
  - 检测同一位置连续拍摄的图像
  - 检测离开工作区域的图像（非植物/土壤）
  - 支持文件复制、移动或软链接

使用示例:
  python enhanced_main.py --source ./images --output ./results
  python enhanced_main.py --config enhanced_config.json --file-operation move
  python enhanced_main.py --preview --source ./images --max-folders 3
  python enhanced_main.py --create-config enhanced_config_template.json
        """
    )
    
    # 基础参数
    parser.add_argument('--source', '-s', type=str, help='源文件夹路径')
    parser.add_argument('--output', '-o', type=str, help='输出文件夹路径')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径')
    
    # 文件操作参数
    parser.add_argument('--file-operation', choices=['copy', 'move', 'symlink'],
                       default='copy', help='文件操作类型')
    
    # 预览和模板参数
    parser.add_argument('--preview', action='store_true', help='预览模式（只分析不处理）')
    parser.add_argument('--max-folders', type=int, default=3, help='预览模式下最大文件夹数')
    parser.add_argument('--create-config', type=str, help='创建配置文件模板')
    
    # 处理参数
    parser.add_argument('--max-workers', type=int, help='最大线程数')
    
    # 日志参数
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    parser.add_argument('--log-file', type=str, help='日志文件路径')
    
    # 阈值调整参数（可选）
    parser.add_argument('--sharpness-threshold', type=float, help='清晰度阈值')
    parser.add_argument('--overexposure-threshold', type=float, help='过曝阈值')
    parser.add_argument('--underexposure-threshold', type=float, help='欠曝阈值')
    parser.add_argument('--similarity-threshold', type=float, help='位置相似度阈值')
    parser.add_argument('--green-threshold', type=float, help='绿色植被阈值')
    
    args = parser.parse_args()
    
    # 创建配置文件模板
    if args.create_config:
        save_config_template(args.create_config)
        return 0
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
        if not config:
            print("配置文件加载失败，使用默认配置")
            config = create_default_config()
    else:
        config = create_default_config()
    
    # 命令行参数覆盖配置文件
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
        
    # 阈值参数覆盖
    if args.sharpness_threshold:
        config['classifier_params']['laplacian_threshold'] = args.sharpness_threshold
    if args.overexposure_threshold:
        config['exposure_params']['overexposure_threshold'] = args.overexposure_threshold
    if args.underexposure_threshold:
        config['exposure_params']['underexposure_threshold'] = args.underexposure_threshold
    if args.similarity_threshold:
        config['position_params']['similarity_threshold'] = args.similarity_threshold
    if args.green_threshold:
        config['work_area_params']['green_threshold'] = args.green_threshold
    
    config['logging']['level'] = args.log_level
    
    # 设置日志
    setup_logging(config['logging']['level'], config['logging'].get('file'))
    logger = logging.getLogger(__name__)
    
    # 验证必要参数
    if not config.get('source_root'):
        logger.error("请指定源文件夹路径 (--source 或在配置文件中设置)")
        return 1
    
    if not config.get('output_root'):
        logger.error("请指定输出文件夹路径 (--output 或在配置文件中设置)")
        return 1
    
    # 检查源文件夹是否存在
    source_path = Path(config['source_root'])
    if not source_path.exists():
        logger.error(f"源文件夹不存在: {source_path}")
        return 1
    
    # 创建批量处理器
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
    
    # 预览模式
    if args.preview:
        logger.info("运行预览模式...")
        preview_results = processor.preview_processing(args.max_folders)
        
        print("\n" + "="*60)
        print("预览结果")
        print("="*60)
        print(f"总共找到文件夹数: {preview_results['total_found_folders']}")
        print(f"预览文件夹数: {preview_results['preview_folder_count']}")
        print("-" * 60)
        
        for folder_info in preview_results['preview_folders']:
            print(f"\n文件夹: {folder_info['folder_name']}")
            print(f"  图像-JSON对数: {folder_info['image_json_pairs']}")
            print(f"  路径: {folder_info['folder_path']}")
            
            if 'sample_analysis' in folder_info:
                print("  样本分析:")
                for sample in folder_info['sample_analysis']:
                    status = "干净" if sample['is_clean'] else f"脏数据 ({', '.join(sample['dirty_reasons'])})"
                    print(f"    - {sample['image']}: {status}")
        
        print("\n" + "="*60)
        return 0
    
    # 正式处理
    logger.info("开始增强批量处理...")
    logger.info(f"文件操作模式: {config['file_operation']}")
    
    try:
        results = processor.run()
        
        logger.info("增强批量处理完成！")
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        return 1
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}", exc_info=True)
        return 1


def run_interactive_mode():
    """交互式运行模式"""
    print("="*60)
    print("增强图像质量分类工具 - 交互模式")
    print("="*60)
    
    # 获取用户输入
    source_root = input("请输入源文件夹路径: ").strip()
    if not source_root:
        print("源文件夹路径不能为空")
        return 1
    
    output_root = input("请输入输出文件夹路径: ").strip()
    if not output_root:
        print("输出文件夹路径不能为空")  
        return 1
    
    # 文件操作类型
    print("\n文件操作类型:")
    print("1. 复制 (copy) - 默认")
    print("2. 移动 (move)")
    print("3. 软链接 (symlink)")
    operation_choice = input("请选择 (1-3, 默认1): ").strip()
    
    operation_map = {'1': 'copy', '2': 'move', '3': 'symlink', '': 'copy'}
    file_operation = operation_map.get(operation_choice, 'copy')
    
    # 询问是否使用默认设置
    use_default = input("\n是否使用默认检测参数? (y/n, 默认y): ").strip().lower()
    
    config = create_default_config()
    config['source_root'] = source_root
    config['output_root'] = output_root
    config['file_operation'] = file_operation
    
    if use_default not in ['y', 'yes', '']:
        # 自定义设置
        print("\n自定义检测参数 (直接回车使用默认值):")
        
        try:
            # 清晰度阈值
            sharp_threshold = input(f"清晰度阈值 (默认{config['classifier_params']['laplacian_threshold']}): ").strip()
            if sharp_threshold:
                config['classifier_params']['laplacian_threshold'] = float(sharp_threshold)
            
            # 曝光阈值
            over_threshold = input(f"过曝阈值 (默认{config['exposure_params']['overexposure_threshold']}): ").strip()
            if over_threshold:
                config['exposure_params']['overexposure_threshold'] = float(over_threshold)
                
            under_threshold = input(f"欠曝阈值 (默认{config['exposure_params']['underexposure_threshold']}): ").strip()
            if under_threshold:
                config['exposure_params']['underexposure_threshold'] = float(under_threshold)
            
            # 位置相似度阈值
            sim_threshold = input(f"位置相似度阈值 (默认{config['position_params']['similarity_threshold']}): ").strip()
            if sim_threshold:
                config['position_params']['similarity_threshold'] = float(sim_threshold)
            
            # 工作区域阈值
            green_threshold = input(f"植被检测阈值 (默认{config['work_area_params']['green_threshold']}): ").strip()
            if green_threshold:
                config['work_area_params']['green_threshold'] = float(green_threshold)
                
            # 线程数
            max_workers = input(f"最大线程数 (默认{config['processing']['max_workers']}): ").strip()
            if max_workers:
                config['processing']['max_workers'] = int(max_workers)
                
        except ValueError as e:
            print(f"输入值错误: {e}")
            return 1
    
    # 设置日志
    setup_logging(config['logging']['level'])
    logger = logging.getLogger(__name__)
    
    # 检查源文件夹
    source_path = Path(source_root)
    if not source_path.exists():
        print(f"源文件夹不存在: {source_path}")
        return 1
    
    # 询问是否先预览
    preview = input("\n是否先预览处理结果? (y/n, 默认n): ").strip().lower()
    
    # 创建处理器
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
        print("\n预览处理结果...")
        preview_results = processor.preview_processing(3)
        
        print(f"\n找到 {preview_results['total_found_folders']} 个时间文件夹")
        for folder_info in preview_results['preview_folders'][:3]:
            print(f"\n文件夹: {folder_info['folder_name']}")
            print(f"  图像对数: {folder_info['image_json_pairs']}")
            
            if 'sample_analysis' in folder_info:
                print("  样本分析:")
                for sample in folder_info['sample_analysis']:
                    status = "干净" if sample['is_clean'] else f"脏数据"
                    if not sample['is_clean']:
                        reasons = {
                            'blurry': '模糊',
                            'overexposed': '过曝',
                            'underexposed': '欠曝',
                            'out_of_work_area': '离开工作区域',
                            'same_position_extra': '同位置重复'
                        }
                        reason_list = [reasons.get(r, r) for r in sample['dirty_reasons']]
                        status += f" ({', '.join(reason_list)})"
                    print(f"    - {sample['image']}: {status}")
        
        continue_process = input("\n是否继续完整处理? (y/n): ").strip().lower()
        if continue_process not in ['y', 'yes']:
            print("已取消处理")
            return 0
    
    # 开始处理
    print(f"\n开始批量处理 (文件操作: {file_operation})...")
    try:
        results = processor.run()
        
        print(f"\n处理完成！结果保存在: {output_root}")
        return 0
        
    except KeyboardInterrupt:
        print("\n用户中断处理")
        return 1
    except Exception as e:
        print(f"\n处理过程中出现错误: {e}")
        return 1


if __name__ == "__main__":
    # 如果没有命令行参数，进入交互模式
    if len(sys.argv) == 1:
        exit_code = run_interactive_mode()
    else:
        exit_code = main()
    
    sys.exit(exit_code)