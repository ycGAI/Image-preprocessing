#!/usr/bin/env python3
"""
图像清晰度分类主程序

使用方法:
    python main.py --source /path/to/source --output /path/to/output
    python main.py --config config.json
    python main.py --preview --source /path/to/source
"""

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
    """设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
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
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return {}


def create_default_config() -> Dict:
    """创建默认配置
    
    Returns:
        默认配置字典
    """
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
    """保存配置文件模板
    
    Args:
        output_path: 输出路径
    """
    config = create_default_config()
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"配置文件模板已保存到: {output_path}")
    except Exception as e:
        print(f"保存配置文件模板失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="图像清晰度分类工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --source ./images --output ./results
  python main.py --config config.json
  python main.py --preview --source ./images --max-folders 3
  python main.py --create-config config_template.json
        """
    )
    
    parser.add_argument('--source', '-s', type=str, help='源文件夹路径')
    parser.add_argument('--output', '-o', type=str, help='输出文件夹路径')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径')
    parser.add_argument('--preview', action='store_true', help='预览模式（只分析不处理）')
    parser.add_argument('--max-folders', type=int, default=3, help='预览模式下最大文件夹数')
    parser.add_argument('--max-workers', type=int, help='最大线程数')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    parser.add_argument('--log-file', type=str, help='日志文件路径')
    parser.add_argument('--create-config', type=str, help='创建配置文件模板')
    parser.add_argument('--no-visual', action='store_true', help='不生成可视化报告')
    parser.add_argument('--no-csv', action='store_true', help='不导出CSV文件')
    
    args = parser.parse_args()
    
    # 创建配置文件模板
    if args.create_config:
        save_config_template(args.create_config)
        return
    
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
    if args.max_workers:
        config['processing']['max_workers'] = args.max_workers
    if args.log_file:
        config['logging']['file'] = args.log_file
    if args.no_visual:
        config['reports']['generate_visual'] = False
    if args.no_csv:
        config['reports']['export_csv'] = False
    
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
    processor = BatchImageProcessor(
        source_root=config['source_root'],
        output_root=config['output_root'],
        classifier_params=config.get('classifier_params', {}),
        max_workers=config['processing']['max_workers'],
        supported_formats=config['processing']['supported_formats']
    )
    
    # 预览模式
    if args.preview:
        logger.info("运行预览模式...")
        preview_results = processor.preview_processing(args.max_folders)
        
        print("\n" + "="*50)
        print("预览结果")
        print("="*50)
        print(f"总共找到文件夹数: {preview_results['total_found_folders']}")
        print(f"预览文件夹数: {preview_results['preview_folder_count']}")
        print("-" * 50)
        
        for folder_info in preview_results['preview_folders']:
            print(f"文件夹: {folder_info['folder_name']}")
            print(f"  图像-JSON对数: {folder_info['image_json_txt_triples']}")
            print(f"  路径: {folder_info['folder_path']}")
            print()
        
        print("="*50)
        return 0
    
    # 正式处理
    logger.info("开始批量处理...")
    try:
        results = processor.run()
        
        # 生成报告
        output_path = Path(config['output_root'])
        report_generator = ReportGenerator(output_path)
        
        # 生成基础报告
        processing_time = results.get('processing_time', 0)
        report = report_generator.generate_processing_report(
            results, 
            processing_time, 
            processor.classifier.get_thresholds()
        )
        
        # 生成详细分析
        detailed_analysis = report_generator.generate_detailed_analysis(results)
        
        # 生成可视化报告
        if config['reports']['generate_visual']:
            try:
                visual_info = report_generator.generate_visual_report(
                    results, 
                    config['reports']['save_plots']
                )
                if visual_info:
                    logger.info("可视化报告生成成功")
            except Exception as e:
                logger.warning(f"生成可视化报告时出错: {e}")
        
        # 导出CSV
        if config['reports']['export_csv']:
            try:
                csv_path = report_generator.export_to_csv(results)
                if csv_path:
                    logger.info(f"CSV文件已导出: {csv_path}")
            except Exception as e:
                logger.warning(f"导出CSV时出错: {e}")
        
        # 打印摘要
        report_generator.print_summary(report)
        
        logger.info("批量处理完成！")
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
    print("图像清晰度分类工具 - 交互模式")
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
    
    # 询问是否使用默认设置
    use_default = input("是否使用默认设置? (y/n, 默认y): ").strip().lower()
    
    config = create_default_config()
    config['source_root'] = source_root
    config['output_root'] = output_root
    
    if use_default not in ['y', 'yes', '']:
        # 自定义设置
        try:
            max_workers = input(f"最大线程数 (默认{config['processing']['max_workers']}): ").strip()
            if max_workers:
                config['processing']['max_workers'] = int(max_workers)
            
            # 分类器阈值设置
            print("\n分类器阈值设置 (直接回车使用默认值):")
            thresholds = config['classifier_params']
            
            for key, default_value in thresholds.items():
                user_input = input(f"{key} (默认{default_value}): ").strip()
                if user_input:
                    thresholds[key] = float(user_input)
                    
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
    preview = input("是否先预览处理结果? (y/n, 默认n): ").strip().lower()
    
    # 创建处理器
    processor = BatchImageProcessor(
        source_root=config['source_root'],
        output_root=config['output_root'],
        classifier_params=config['classifier_params'],
        max_workers=config['processing']['max_workers'],
        supported_formats=config['processing']['supported_formats']
    )
    
    if preview in ['y', 'yes']:
        print("\n预览处理结果...")
        preview_results = processor.preview_processing(3)
        
        print(f"\n找到 {preview_results['total_found_folders']} 个时间文件夹")
        for folder_info in preview_results['preview_folders']:
            print(f"- {folder_info['folder_name']}: {folder_info['image_json_txt_triples']} 个图像对")
        
        continue_process = input("\n是否继续完整处理? (y/n): ").strip().lower()
        if continue_process not in ['y', 'yes']:
            print("已取消处理")
            return 0
    
    # 开始处理
    print("\n开始批量处理...")
    try:
        results = processor.run()
        
        # 生成报告
        output_path = Path(config['output_root'])
        report_generator = ReportGenerator(output_path)
        
        processing_time = results.get('processing_time', 0)
        report = report_generator.generate_processing_report(
            results, 
            processing_time, 
            processor.classifier.get_thresholds()
        )
        
        # 打印摘要
        report_generator.print_summary(report)
        
        print(f"\n处理完成！结果保存在: {output_path}")
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