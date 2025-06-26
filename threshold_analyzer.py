#!/usr/bin/env python3
"""
修正版工作阈值分析器 - 生成兼容格式的配置文件

主要修正：
1. 生成与原始格式完全兼容的配置文件
2. 保留所有必需字段
3. 支持自定义路径
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import cv2
import numpy as np
from pathlib import Path
import random
import json
import argparse
import sys
import time
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats, optimize
from sklearn.model_selection import train_test_split
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_image_batch(args):
    """批量分析图像的工作函数"""
    image_paths, max_size = args
    results = []
    
    for image_path in image_paths:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                results.append(None)
                continue
            
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            gray_float = gray.astype(np.float64)
            
            # Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = float(laplacian.var())
            
            # Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            sobel_var = float(sobel_mag.var())
            
            # Brenner
            diff = np.diff(gray_float, axis=1)
            brenner = float(np.sum(diff**2))
            
            # Tenengrad
            threshold = np.mean(sobel_mag)
            tenengrad = float(np.sum(sobel_mag[sobel_mag > threshold]**2))
            
            results.append({
                'path': str(image_path),
                'metrics': {
                    'laplacian': laplacian_var,
                    'sobel': sobel_var,
                    'brenner': brenner,
                    'tenengrad': tenengrad,
                    'image_size': h * w,
                    'aspect_ratio': w / h
                },
                'folder': Path(image_path).parent.name
            })
            
        except Exception as e:
            logger.error(f"分析图像失败 {image_path}: {e}")
            results.append(None)
    
    return results


class FixedWorkingAnalyzer:
    """修正版工作阈值分析器"""
    
    def __init__(self, target_blur_rate: float = 0.1, n_workers: int = None):
        self.target_blur_rate = target_blur_rate
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        self.methods = ['laplacian', 'sobel', 'brenner', 'tenengrad']
        self.sample_data = None
        self.train_data = None
        self.val_data = None
        self.n_workers = n_workers or cpu_count()
        
        # 配置文件模板
        self.config_template = {
            "source_root": "/media/gyc/Backup Plus5/gyc/ATB_data/raw_data",
            "output_root": "/media/gyc/Backup Plus5/gyc/ATB_data/output_test",
            "processing": {
                "max_workers": 8,
                "supported_formats": [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tiff",
                    ".webp"
                ]
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
        
    def collect_and_analyze_samples(self, folder_path: Path, sample_size: int = 500, 
                                  batch_size: int = 50, max_image_size: int = 1024) -> Dict:
        """收集并分析样本数据"""
        start_time = time.time()
        logger.info(f"从 {folder_path} 收集样本数据...")
        logger.info(f"使用 {self.n_workers} 个工作进程")
        
        # 收集图像文件
        image_files = []
        for fmt in self.supported_formats:
            image_files.extend(folder_path.glob(f'**/*{fmt}'))
            image_files.extend(folder_path.glob(f'**/*{fmt.upper()}'))
        
        if len(image_files) > sample_size:
            image_files = random.sample(image_files, sample_size)
        
        logger.info(f"分析 {len(image_files)} 个图像文件...")
        
        # 分批处理
        batches = []
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            batches.append((batch, max_image_size))
        
        # 多进程处理
        sample_data = []
        failed_count = 0
        
        with Pool(self.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(analyze_image_batch, batches),
                total=len(batches),
                desc="处理批次"
            ))
        
        for batch_results in results:
            for result in batch_results:
                if result:
                    sample_data.append(result)
                else:
                    failed_count += 1
        
        self.sample_data = sample_data
        elapsed_time = time.time() - start_time
        logger.info(f"成功分析 {len(sample_data)} 个图像，失败 {failed_count} 个")
        logger.info(f"总耗时: {elapsed_time:.2f} 秒 ({len(sample_data)/elapsed_time:.1f} 图像/秒)")
        
        # 分析数据分布
        distribution_analysis = self._analyze_distribution(sample_data)
        
        return distribution_analysis
    
    def _analyze_distribution(self, sample_data: List[Dict]) -> Dict:
        """分析数据分布"""
        analysis = {
            'sample_count': len(sample_data),
            'methods': {}
        }
        
        for method in self.methods:
            values = np.array([s['metrics'][method] for s in sample_data])
            
            percentiles_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
            percentile_values = np.percentile(values, percentiles_list)
            
            analysis['methods'][method] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'percentiles': {
                    int(p): float(v) for p, v in zip(percentiles_list, percentile_values)
                }
            }
        
        return analysis
    
    def visualize_distribution_safe(self, save_path: Optional[Path] = None):
        """安全的可视化（只保存，不显示）"""
        if not self.sample_data:
            logger.warning("没有样本数据可视化")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            for idx, method in enumerate(self.methods):
                values = [s['metrics'][method] for s in self.sample_data]
                
                ax = axes[idx]
                ax.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                
                # 添加百分位线
                for p, color, style in [(50, 'blue', '-'), (70, 'green', '--'), (90, 'red', ':')]:
                    val = np.percentile(values, p)
                    ax.axvline(val, color=color, linestyle=style, alpha=0.7, 
                              label=f'P{p}: {val:.0f}')
                
                ax.set_xlabel(f'{method}')
                ax.set_ylabel('Count')
                ax.set_title(f'{method} Distribution')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Sharpness Metrics Distribution', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                output_file = save_path / 'distribution_analysis.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                logger.info(f"分布图已保存到: {output_file}")
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"可视化失败: {e}")
            logger.info("继续执行其他步骤...")
    
    def split_data(self, test_size: float = 0.3):
        """分割数据"""
        if not self.sample_data:
            raise ValueError("没有样本数据")
        
        indices = np.arange(len(self.sample_data))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        self.train_data = [self.sample_data[i] for i in train_idx]
        self.val_data = [self.sample_data[i] for i in val_idx]
        
        logger.info(f"数据分割完成：训练集 {len(self.train_data)}，验证集 {len(self.val_data)}")
    
    def _simulate_classification(self, thresholds: Dict[str, float], 
                               data: Optional[List[Dict]] = None) -> float:
        """模拟分类过程"""
        if data is None:
            data = self.sample_data
        
        if not data:
            raise ValueError("没有数据进行模拟")
        
        blur_count = 0
        for sample in data:
            votes = sum(1 for m in self.methods if sample['metrics'][m] > thresholds[m])
            if votes <= len(self.methods) / 2:
                blur_count += 1
        
        return blur_count / len(data)
    
    def optimize_thresholds_simple(self, data: Optional[List[Dict]] = None) -> Dict[str, float]:
        """简单但有效的阈值优化 - 修正版"""
        if data is None:
            data = self.train_data if self.train_data else self.sample_data
        
        if not data:
            raise ValueError("没有数据用于优化")
        
        logger.info("开始优化阈值...")
        
        # 收集每个方法的值
        method_values = {}
        for method in self.methods:
            method_values[method] = np.array([d['metrics'][method] for d in data])
        
        # 修正：如果目标是10%模糊率，意味着90%应该被判定为清晰
        # 因此初始阈值应该设置在较低的百分位
        # 使用第10百分位作为初始值（90%的样本值会高于这个阈值）
        target_clear_rate = 1 - self.target_blur_rate  # 0.9
        initial_percentile = (1 - target_clear_rate) * 100  # 10
        
        current_thresholds = {}
        for method in self.methods:
            current_thresholds[method] = np.percentile(method_values[method], initial_percentile)
        
        logger.info(f"初始阈值设置在第{initial_percentile:.0f}百分位")
        
        # 迭代优化
        best_thresholds = current_thresholds.copy()
        best_error = float('inf')
        
        # 修正：搜索范围应该在较低的百分位区间
        # 对于10%模糊率，搜索范围大约在5-30百分位
        percentile_range = range(5, 31, 1)
        
        logger.info("搜索最佳百分位组合...")
        for base_percentile in tqdm(percentile_range, desc="优化进度"):
            # 尝试对所有方法使用相同的百分位
            test_thresholds = {}
            for method in self.methods:
                test_thresholds[method] = np.percentile(method_values[method], base_percentile)
            
            blur_rate = self._simulate_classification(test_thresholds, data)
            error = abs(blur_rate - self.target_blur_rate)
            
            if error < best_error:
                best_error = error
                best_thresholds = test_thresholds.copy()
                
                if error < 0.01:  # 误差小于1%
                    logger.info(f"找到满意的阈值（误差 {error:.3f}，百分位 {base_percentile}）")
                    break
        
        # 微调每个方法
        logger.info("微调各个方法的阈值...")
        for method in self.methods:
            current_value = best_thresholds[method]
            best_value = current_value
            local_best_error = best_error
            
            # 在当前值的±20%范围内搜索
            test_range = np.linspace(current_value * 0.8, current_value * 1.2, 21)
            
            for test_value in test_range:
                test_thresholds = best_thresholds.copy()
                test_thresholds[method] = test_value
                
                blur_rate = self._simulate_classification(test_thresholds, data)
                error = abs(blur_rate - self.target_blur_rate)
                
                if error < local_best_error:
                    local_best_error = error
                    best_value = test_value
            
            best_thresholds[method] = best_value
            best_error = local_best_error
        
        # 验证最终结果
        final_blur_rate = self._simulate_classification(best_thresholds, data)
        logger.info(f"优化完成，最小误差: {best_error:.4f}")
        logger.info(f"最终模糊率: {final_blur_rate:.1%} (目标: {self.target_blur_rate:.1%})")
        
        return best_thresholds
    
    def validate_thresholds(self, thresholds: Dict[str, float]) -> Dict:
        """验证阈值"""
        val_data = self.val_data if self.val_data else self.sample_data
        
        val_blur_rate = self._simulate_classification(thresholds, val_data)
        
        return {
            'validation_blur_rate': val_blur_rate,
            'target_blur_rate': self.target_blur_rate,
            'error': abs(val_blur_rate - self.target_blur_rate)
        }
    
    def generate_report(self, optimized_thresholds: Dict[str, float],
                       distribution_analysis: Dict,
                       validation_results: Dict,
                       output_path: Path,
                       source_root: Optional[str] = None,
                       output_root: Optional[str] = None):
        """生成报告和兼容格式的配置文件"""
        
        # 生成兼容格式的配置文件
        config_path = output_path / 'optimized_config.json'
        
        # 创建完整的配置
        config = {
            "source_root": source_root or self.config_template["source_root"],
            "output_root": output_root or self.config_template["output_root"],
            "classifier_params": {
                "laplacian_threshold": float(optimized_thresholds.get('laplacian', 100.0)),
                "sobel_threshold": float(optimized_thresholds.get('sobel', 50.0)),
                "brenner_threshold": float(optimized_thresholds.get('brenner', 1000.0)),
                "tenengrad_threshold": float(optimized_thresholds.get('tenengrad', 500.0)),
                "variance_threshold": 50.0  # 保持原有的variance_threshold
            },
            "processing": self.config_template["processing"],
            "logging": self.config_template["logging"],
            "reports": self.config_template["reports"]
        }
        
        # 添加优化信息（作为注释或额外字段）
        config["_optimization_info"] = {
            "target_blur_rate": self.target_blur_rate,
            "achieved_blur_rate": validation_results.get('validation_blur_rate', 0),
            "validation_error": validation_results.get('error', 0),
            "sample_size": len(self.sample_data) if self.sample_data else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化配置已保存到: {config_path}")
        
        # 生成详细报告
        report_path = output_path / 'optimization_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("阈值优化报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"样本数量: {len(self.sample_data)}\n")
            f.write(f"目标模糊率: {self.target_blur_rate:.1%}\n")
            f.write(f"实现模糊率: {validation_results.get('validation_blur_rate', 0):.1%}\n")
            f.write(f"误差: {validation_results.get('error', 0):.1%}\n\n")
            
            f.write("优化后的阈值:\n")
            for method, threshold in optimized_thresholds.items():
                if method in distribution_analysis['methods']:
                    stats = distribution_analysis['methods'][method]
                    # 找到最接近的百分位
                    percentile = "未知"
                    for p in sorted(stats['percentiles'].keys()):
                        if threshold <= stats['percentiles'][p]:
                            percentile = f"约第{p}百分位"
                            break
                    else:
                        percentile = ">99百分位"
                    
                    f.write(f"  {method}_threshold: {threshold:.2f} ({percentile})\n")
            
            f.write("\n数据分布统计:\n")
            for method, stats in distribution_analysis['methods'].items():
                f.write(f"\n{method}:\n")
                f.write(f"  均值: {stats['mean']:.2f}\n")
                f.write(f"  标准差: {stats['std']:.2f}\n")
                f.write(f"  中位数: {stats['median']:.2f}\n")
                f.write(f"  范围: [{stats['min']:.2f}, {stats['max']:.2f}]\n")
        
        logger.info(f"优化报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="修正版工作阈值分析器")
    
    parser.add_argument('folder', help='样本图像文件夹路径')
    parser.add_argument('--target-blur-rate', type=float, default=0.1,
                       help='目标模糊率 (默认0.1表示10%%)')
    parser.add_argument('--sample-size', type=int, default=500,
                       help='分析的样本数量')
    parser.add_argument('--workers', type=int, default=None,
                       help='工作进程数')
    parser.add_argument('--validate', action='store_true',
                       help='使用验证集验证结果')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='生成并保存可视化图表')
    parser.add_argument('--source-root', type=str, 
                       default="/media/gyc/Backup Plus5/gyc/ATB_data/raw_data",
                       help='源文件根目录路径')
    parser.add_argument('--output-root', type=str,
                       default="/media/gyc/Backup Plus5/gyc/ATB_data/output_test",
                       help='输出文件根目录路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器
    analyzer = FixedWorkingAnalyzer(
        target_blur_rate=args.target_blur_rate,
        n_workers=args.workers
    )
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        logger.error(f"文件夹不存在: {folder_path}")
        return 1
    
    # 记录总时间
    total_start = time.time()
    
    # 步骤1：收集和分析数据
    print("\n" + "="*60)
    print("步骤1：数据收集和分布分析")
    print("="*60)
    # import ipdb; ipdb.set_trace()  
    distribution_analysis = analyzer.collect_and_analyze_samples(
        folder_path, 
        sample_size=args.sample_size
    )
    
    # 打印分布摘要
    print("\n数据分布摘要:")
    for method, stats in distribution_analysis['methods'].items():
        print(f"\n{method}:")
        print(f"  均值: {stats['mean']:.2f} (±{stats['std']:.2f})")
        print(f"  中位数: {stats['median']:.2f}")
        print(f"  P70: {stats['percentiles'][70]:.2f}")
    
    # 可视化（安全模式）
    if args.visualize:
        print("\n生成可视化图表...")
        analyzer.visualize_distribution_safe(save_path=args.output_dir)
    
    # 步骤2：数据分割
    if args.validate:
        print("\n" + "="*60)
        print("步骤2：数据分割")
        print("="*60)
        analyzer.split_data(test_size=0.3)
    
    # 步骤3：阈值优化
    print("\n" + "="*60)
    print("步骤3：阈值优化")
    print("="*60)
    
    # 初始阈值
    initial_thresholds = {}
    for method, stats in distribution_analysis['methods'].items():
        initial_thresholds[method] = stats['percentiles'][70]
    
    initial_blur_rate = analyzer._simulate_classification(initial_thresholds)
    print(f"\n初始模糊率 (P70): {initial_blur_rate:.1%}")
    print(f"目标模糊率: {args.target_blur_rate:.1%}")
    
    # 优化
    opt_start = time.time()
    optimized_thresholds = analyzer.optimize_thresholds_simple()
    opt_time = time.time() - opt_start
    
    # 显示优化结果
    optimized_blur_rate = analyzer._simulate_classification(optimized_thresholds)
    print(f"\n优化完成（耗时 {opt_time:.1f} 秒）")
    print(f"优化后的模糊率: {optimized_blur_rate:.1%}")
    print("\n优化后的阈值:")
    for method, threshold in optimized_thresholds.items():
        initial = initial_thresholds[method]
        change = (threshold - initial) / initial * 100
        print(f"  {method}: {threshold:.2f} (变化: {change:+.1f}%)")
    
    # 步骤4：验证
    validation_results = {}
    if args.validate:
        print("\n" + "="*60)
        print("步骤4：验证集验证")
        print("="*60)
        
        validation_results = analyzer.validate_thresholds(optimized_thresholds)
        print(f"\n验证集模糊率: {validation_results['validation_blur_rate']:.1%}")
        print(f"验证误差: {validation_results['error']:.1%}")
    else:
        validation_results = {
            'validation_blur_rate': optimized_blur_rate,
            'error': abs(optimized_blur_rate - args.target_blur_rate)
        }
    
    # 步骤5：生成报告
    print("\n" + "="*60)
    print("步骤5：生成报告")
    print("="*60)
    
    analyzer.generate_report(
        optimized_thresholds,
        distribution_analysis,
        validation_results,
        args.output_dir,
        source_root=args.source_root,
        output_root=args.output_root
    )
    
    # 总结
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print(f"优化完成！总耗时: {total_time:.1f} 秒")
    print(f"配置文件已保存到: {args.output_dir / 'optimized_config.json'}")
    print("\n建议使用以下命令运行主程序:")
    print(f"python main.py --config {args.output_dir / 'optimized_config.json'}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())