#!/usr/bin/env python3
"""
阈值分析和调整工具

用于分析图像数据集的清晰度指标分布，帮助确定合适的阈值
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

from sharpness_classifier import ImageSharpnessClassifier
from file_utils import FileUtils

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class ThresholdAnalyzer:
    """阈值分析器"""
    
    def __init__(self, supported_formats=None):
        """初始化分析器"""
        self.file_utils = FileUtils(supported_formats)
        # 使用较低的初始阈值来计算所有指标
        self.classifier = ImageSharpnessClassifier(
            laplacian_threshold=1.0,
            sobel_threshold=1.0,
            brenner_threshold=1.0,
            tenengrad_threshold=1.0
        )
        
    def analyze_folder(self, folder_path: Path, max_images: int = 100) -> Dict:
        """分析文件夹中图像的清晰度指标分布
        
        Args:
            folder_path: 文件夹路径
            max_images: 最大分析图像数（防止内存溢出）
            
        Returns:
            分析结果字典
        """
        logger.info(f"开始分析文件夹: {folder_path}")
        
        # 找到所有图像文件
        image_files = self.file_utils.find_all_images(folder_path)
        
        if not image_files:
            logger.warning(f"文件夹 {folder_path} 中没有找到图像文件")
            return {}
        
        # 限制分析的图像数量
        if len(image_files) > max_images:
            logger.info(f"图像数量 {len(image_files)} 超过限制 {max_images}，随机选择")
            import random
            image_files = random.sample(image_files, max_images)
        
        metrics_data = {
            'laplacian': [],
            'sobel': [],
            'brenner': [],
            'tenengrad': [],
            'image_paths': []
        }
        
        failed_count = 0
        
        # 分析每个图像
        for image_path in tqdm(image_files, desc="分析图像"):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    failed_count += 1
                    continue
                
                # 计算所有清晰度指标
                metrics = self.classifier.calculate_all_metrics(image)
                
                metrics_data['laplacian'].append(metrics['laplacian'])
                metrics_data['sobel'].append(metrics['sobel'])
                metrics_data['brenner'].append(metrics['brenner'])
                metrics_data['tenengrad'].append(metrics['tenengrad'])
                metrics_data['image_paths'].append(str(image_path))
                
            except Exception as e:
                logger.error(f"分析图像 {image_path} 失败: {e}")
                failed_count += 1
        
        # 计算统计信息
        analysis_result = {
            'folder_path': str(folder_path),
            'total_images': len(image_files),
            'analyzed_images': len(metrics_data['laplacian']),
            'failed_images': failed_count,
            'metrics_data': metrics_data,
            'statistics': self._calculate_statistics(metrics_data)
        }
        
        logger.info(f"分析完成: {analysis_result['analyzed_images']} 个图像")
        return analysis_result
    
    def _calculate_statistics(self, metrics_data: Dict) -> Dict:
        """计算统计信息"""
        stats = {}
        
        for metric_name in ['laplacian', 'sobel', 'brenner', 'tenengrad']:
            values = np.array(metrics_data[metric_name])
            if len(values) == 0:
                continue
                
            stats[metric_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'q90': float(np.percentile(values, 90)),
                'q95': float(np.percentile(values, 95))
            }
        
        return stats
    
    def suggest_thresholds(self, analysis_result: Dict, 
                          sharp_ratio_target: float = 0.7) -> Dict:
        """根据分析结果建议阈值
        
        Args:
            analysis_result: 分析结果
            sharp_ratio_target: 目标清晰图像比例
            
        Returns:
            建议的阈值
        """
        stats = analysis_result['statistics']
        suggested_thresholds = {}
        
        # 根据目标清晰度比例计算阈值
        # 如果目标是70%的图像为清晰，则使用30%分位数作为阈值
        percentile = (1 - sharp_ratio_target) * 100
        
        for metric_name in ['laplacian', 'sobel', 'brenner', 'tenengrad']:
            if metric_name in stats:
                values = np.array(analysis_result['metrics_data'][metric_name])
                threshold = np.percentile(values, percentile)
                suggested_thresholds[metric_name] = float(threshold)
        
        return suggested_thresholds
    
    def plot_distributions(self, analysis_result: Dict, 
                          current_thresholds: Dict = None,
                          suggested_thresholds: Dict = None,
                          save_path: Path = None) -> None:
        """绘制指标分布图
        
        Args:
            analysis_result: 分析结果
            current_thresholds: 当前阈值
            suggested_thresholds: 建议阈值
            save_path: 保存路径
        """
        metrics_data = analysis_result['metrics_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_names = ['laplacian', 'sobel', 'brenner', 'tenengrad']
        metric_titles = ['拉普拉斯方差', 'Sobel方差', 'Brenner梯度', 'Tenengrad方差']
        
        for i, (metric_name, title) in enumerate(zip(metric_names, metric_titles)):
            if metric_name not in metrics_data or not metrics_data[metric_name]:
                continue
                
            ax = axes[i]
            values = metrics_data[metric_name]
            
            # 绘制直方图
            ax.hist(values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # 添加当前阈值线
            if current_thresholds and metric_name in current_thresholds:
                ax.axvline(current_thresholds[metric_name], 
                          color='red', linestyle='--', linewidth=2, 
                          label=f'当前阈值: {current_thresholds[metric_name]:.1f}')
            
            # 添加建议阈值线
            if suggested_thresholds and metric_name in suggested_thresholds:
                ax.axvline(suggested_thresholds[metric_name], 
                          color='green', linestyle='-', linewidth=2,
                          label=f'建议阈值: {suggested_thresholds[metric_name]:.1f}')
            
            ax.set_title(f'{title} 分布')
            ax.set_xlabel('指标值')
            ax.set_ylabel('频数')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分布图已保存到: {save_path}")
        
        plt.show()
    
    def generate_threshold_report(self, analysis_result: Dict, 
                                 current_thresholds: Dict,
                                 suggested_thresholds: Dict,
                                 output_path: Path) -> None:
        """生成阈值分析报告
        
        Args:
            analysis_result: 分析结果
            current_thresholds: 当前阈值
            suggested_thresholds: 建议阈值
            output_path: 输出路径
        """
        report = {
            'analysis_summary': {
                'folder_path': analysis_result['folder_path'],
                'total_images': analysis_result['total_images'],
                'analyzed_images': analysis_result['analyzed_images'],
                'failed_images': analysis_result['failed_images']
            },
            'current_thresholds': current_thresholds,
            'suggested_thresholds': suggested_thresholds,
            'statistics': analysis_result['statistics'],
            'threshold_comparison': self._compare_thresholds(
                analysis_result, current_thresholds, suggested_thresholds
            )
        }
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"阈值分析报告已保存到: {output_path}")
        
        # 打印摘要
        self._print_threshold_summary(report)
    
    def _compare_thresholds(self, analysis_result: Dict, 
                           current_thresholds: Dict, 
                           suggested_thresholds: Dict) -> Dict:
        """比较当前阈值和建议阈值的效果"""
        metrics_data = analysis_result['metrics_data']
        comparison = {}
        
        for metric_name in ['laplacian', 'sobel', 'brenner', 'tenengrad']:
            if metric_name not in metrics_data or not metrics_data[metric_name]:
                continue
                
            values = np.array(metrics_data[metric_name])
            
            # 当前阈值的分类结果
            current_threshold = current_thresholds.get(metric_name, 0)
            current_sharp_count = np.sum(values > current_threshold)
            current_sharp_ratio = current_sharp_count / len(values)
            
            # 建议阈值的分类结果
            suggested_threshold = suggested_thresholds.get(metric_name, 0)
            suggested_sharp_count = np.sum(values > suggested_threshold)
            suggested_sharp_ratio = suggested_sharp_count / len(values)
            
            comparison[metric_name] = {
                'current_sharp_ratio': float(current_sharp_ratio),
                'suggested_sharp_ratio': float(suggested_sharp_ratio),
                'threshold_change': float(suggested_threshold - current_threshold),
                'threshold_change_ratio': float((suggested_threshold - current_threshold) / current_threshold) if current_threshold > 0 else float('inf')
            }
        
        return comparison
    
    def _print_threshold_summary(self, report: Dict):
        """打印阈值摘要"""
        print("\n" + "="*60)
        print("阈值分析报告摘要")
        print("="*60)
        
        summary = report['analysis_summary']
        print(f"分析文件夹: {summary['folder_path']}")
        print(f"总图像数: {summary['total_images']}")
        print(f"成功分析: {summary['analyzed_images']}")
        print(f"失败图像: {summary['failed_images']}")
        
        print("\n当前阈值 vs 建议阈值:")
        print("-" * 60)
        
        current = report['current_thresholds']
        suggested = report['suggested_thresholds']
        comparison = report['threshold_comparison']
        
        for metric in ['laplacian', 'sobel', 'brenner', 'tenengrad']:
            if metric in current and metric in suggested:
                current_val = current[metric]
                suggested_val = suggested[metric]
                comp = comparison[metric]
                
                print(f"{metric:12} | 当前: {current_val:8.1f} | 建议: {suggested_val:8.1f} | "
                      f"变化: {comp['threshold_change']:+8.1f} | "
                      f"清晰比例: {comp['current_sharp_ratio']:.1%} → {comp['suggested_sharp_ratio']:.1%}")
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图像清晰度阈值分析工具")
    parser.add_argument('--source', '-s', required=True, help='源文件夹路径')
    parser.add_argument('--output', '-o', default='./threshold_analysis', help='输出目录')
    parser.add_argument('--max-images', type=int, default=100, help='最大分析图像数')
    parser.add_argument('--target-ratio', type=float, default=0.7, help='目标清晰图像比例')
    parser.add_argument('--current-config', help='当前配置文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器
    analyzer = ThresholdAnalyzer()
    
    # 分析文件夹
    source_path = Path(args.source)
    if source_path.is_dir():
        analysis_result = analyzer.analyze_folder(source_path, args.max_images)
    else:
        logger.error(f"源路径不是有效的文件夹: {source_path}")
        return 1
    
    if not analysis_result:
        logger.error("分析失败")
        return 1
    
    # 获取当前阈值
    current_thresholds = {
        'laplacian': 100.0,
        'sobel': 50.0,
        'brenner': 1000.0,
        'tenengrad': 500.0
    }
    
    if args.current_config:
        try:
            with open(args.current_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'classifier_params' in config:
                    current_thresholds.update({
                        'laplacian': config['classifier_params'].get('laplacian_threshold', 100.0),
                        'sobel': config['classifier_params'].get('sobel_threshold', 50.0),
                        'brenner': config['classifier_params'].get('brenner_threshold', 1000.0),
                        'tenengrad': config['classifier_params'].get('tenengrad_threshold', 500.0)
                    })
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}")
    
    # 建议新阈值
    suggested_thresholds = analyzer.suggest_thresholds(analysis_result, args.target_ratio)
    
    # 生成报告
    report_path = output_dir / 'threshold_analysis_report.json'
    analyzer.generate_threshold_report(analysis_result, current_thresholds, 
                                     suggested_thresholds, report_path)
    
    # 绘制分布图
    plot_path = output_dir / 'metrics_distribution.png'
    analyzer.plot_distributions(analysis_result, current_thresholds, 
                               suggested_thresholds, plot_path)
    
    # 生成新配置文件
    new_config = {
        "classifier_params": {
            "laplacian_threshold": suggested_thresholds.get('laplacian', 100.0),
            "sobel_threshold": suggested_thresholds.get('sobel', 50.0),
            "brenner_threshold": suggested_thresholds.get('brenner', 1000.0),
            "tenengrad_threshold": suggested_thresholds.get('tenengrad', 500.0)
        }
    }
    
    config_path = output_dir / 'suggested_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(new_config, f, ensure_ascii=False, indent=2)
    
    print(f"\n建议的配置文件已保存到: {config_path}")
    print("请将建议的阈值复制到你的配置文件中。")
    
    return 0


if __name__ == "__main__":
    exit(main())