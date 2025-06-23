#!/usr/bin/env python3
"""
智能阈值优化器 - 通过模拟投票机制找到最佳阈值组合

使用方法:
    python smart_threshold_optimizer.py /path/to/sample/folder --target-blur-rate 0.1
    python smart_threshold_optimizer.py /path/to/sample/folder --target-blur-rate 0.1 --iterations 20
"""

import cv2
import numpy as np
from pathlib import Path
import random
import json
import argparse
import sys
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
from scipy import optimize
import matplotlib.pyplot as plt
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartThresholdOptimizer:
    """智能阈值优化器，考虑投票机制的影响"""
    
    def __init__(self, target_blur_rate: float = 0.1):
        """
        初始化优化器
        
        Args:
            target_blur_rate: 目标模糊率（例如0.1表示10%的图像应该被判定为模糊）
        """
        self.target_blur_rate = target_blur_rate
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        self.methods = ['laplacian', 'sobel', 'brenner', 'tenengrad']
        self.sample_data = None
        
    def collect_sample_data(self, folder_path: Path, sample_size: int = 200) -> List[Dict]:
        """收集样本数据"""
        logger.info(f"从 {folder_path} 收集样本数据...")
        
        # 收集图像文件
        image_files = []
        for fmt in self.supported_formats:
            image_files.extend(folder_path.glob(f'**/*{fmt}'))
            image_files.extend(folder_path.glob(f'**/*{fmt.upper()}'))
        
        # 随机采样
        if len(image_files) > sample_size:
            image_files = random.sample(image_files, sample_size)
        
        logger.info(f"分析 {len(image_files)} 个图像文件...")
        
        # 分析每个图像
        sample_data = []
        for img_path in tqdm(image_files, desc="分析图像"):
            metrics = self._analyze_image(img_path)
            if metrics:
                sample_data.append({
                    'path': img_path,
                    'metrics': metrics
                })
        
        self.sample_data = sample_data
        logger.info(f"成功分析 {len(sample_data)} 个图像")
        return sample_data
    
    def _analyze_image(self, image_path: Path, max_size: int = 1024) -> Dict[str, float]:
        """分析单个图像"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # 缩放
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 转灰度
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 计算指标
            metrics = {}
            
            # Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            metrics['laplacian'] = float(laplacian.var())
            
            # Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            metrics['sobel'] = float(sobel.var())
            
            # Brenner
            diff = np.diff(gray.astype(np.float64), axis=1)
            metrics['brenner'] = float(np.sum(diff**2))
            
            # Tenengrad
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            threshold = np.mean(gradient_magnitude)
            metrics['tenengrad'] = float(np.sum(gradient_magnitude[gradient_magnitude > threshold]**2))
            
            return metrics
            
        except Exception as e:
            logger.error(f"分析图像失败 {image_path}: {e}")
            return None
    
    def simulate_classification(self, thresholds: Dict[str, float]) -> float:
        """
        模拟分类过程，返回模糊率
        
        Args:
            thresholds: 各方法的阈值
            
        Returns:
            模糊图像的比例
        """
        if not self.sample_data:
            raise ValueError("没有样本数据，请先调用 collect_sample_data")
        
        blur_count = 0
        
        for sample in self.sample_data:
            # 投票
            votes = 0
            for method in self.methods:
                if sample['metrics'][method] > thresholds[method]:
                    votes += 1
            
            # 如果不超过半数投票认为清晰，则判定为模糊
            if votes <= len(self.methods) / 2:
                blur_count += 1
        
        blur_rate = blur_count / len(self.sample_data)
        return blur_rate
    
    def optimize_thresholds_grid_search(self, initial_stats: Dict) -> Dict[str, float]:
        """
        使用网格搜索优化阈值
        
        Args:
            initial_stats: 初始统计信息
            
        Returns:
            优化后的阈值
        """
        logger.info("开始网格搜索优化...")
        
        # 为每个方法生成候选阈值
        threshold_candidates = {}
        for method in self.methods:
            if method in initial_stats:
                percentiles = initial_stats[method]['percentiles']
                # 从50到95百分位数中选择候选值
                candidates = []
                for p in range(50, 96, 5):
                    if str(p) in percentiles:
                        candidates.append(percentiles[str(p)])
                threshold_candidates[method] = candidates
        
        # 记录最佳结果
        best_thresholds = None
        best_error = float('inf')
        
        # 网格搜索（简化版，随机采样组合）
        max_iterations = 1000
        for i in tqdm(range(max_iterations), desc="搜索最佳阈值组合"):
            # 随机选择一个阈值组合
            current_thresholds = {}
            for method, candidates in threshold_candidates.items():
                current_thresholds[method] = random.choice(candidates)
            
            # 模拟分类
            blur_rate = self.simulate_classification(current_thresholds)
            
            # 计算误差
            error = abs(blur_rate - self.target_blur_rate)
            
            if error < best_error:
                best_error = error
                best_thresholds = current_thresholds.copy()
                
                if error < 0.01:  # 误差小于1%就认为足够好
                    logger.info(f"找到满意的阈值组合，模糊率: {blur_rate:.1%}")
                    break
        
        return best_thresholds
    
    def optimize_thresholds_smart(self, initial_stats: Dict) -> Dict[str, float]:
        """
        使用智能算法优化阈值（二分搜索）
        
        Args:
            initial_stats: 初始统计信息
            
        Returns:
            优化后的阈值
        """
        logger.info("使用智能算法优化阈值...")
        
        # 初始阈值：使用各方法的70百分位数
        current_thresholds = {}
        for method in self.methods:
            if method in initial_stats and 'percentiles' in initial_stats[method]:
                current_thresholds[method] = initial_stats[method]['percentiles']['70']
        
        # 迭代优化
        max_iterations = 50
        learning_rate = 0.1
        
        for iteration in range(max_iterations):
            # 当前模糊率
            current_blur_rate = self.simulate_classification(current_thresholds)
            error = current_blur_rate - self.target_blur_rate
            
            logger.info(f"迭代 {iteration + 1}: 当前模糊率 {current_blur_rate:.1%}, "
                       f"目标 {self.target_blur_rate:.1%}, 误差 {error:.1%}")
            
            # 如果足够接近目标，停止
            if abs(error) < 0.01:
                logger.info("达到目标模糊率！")
                break
            
            # 调整阈值
            # 如果模糊率太高，需要降低阈值（让更多图像被判定为清晰）
            # 如果模糊率太低，需要提高阈值（让更多图像被判定为模糊）
            adjustment_factor = 1.0 - error * learning_rate
            
            for method in self.methods:
                # 根据各方法的敏感度进行不同程度的调整
                method_sensitivity = self._calculate_method_sensitivity(method, current_thresholds)
                current_thresholds[method] *= adjustment_factor ** method_sensitivity
        
        return current_thresholds
    
    def _calculate_method_sensitivity(self, method: str, current_thresholds: Dict[str, float]) -> float:
        """计算某个方法对结果的敏感度"""
        # 临时改变该方法的阈值，看对结果的影响
        original_threshold = current_thresholds[method]
        
        # 增加10%
        current_thresholds[method] = original_threshold * 1.1
        blur_rate_high = self.simulate_classification(current_thresholds)
        
        # 减少10%
        current_thresholds[method] = original_threshold * 0.9
        blur_rate_low = self.simulate_classification(current_thresholds)
        
        # 恢复原值
        current_thresholds[method] = original_threshold
        
        # 计算敏感度
        sensitivity = abs(blur_rate_high - blur_rate_low) / 0.2
        return max(0.5, min(2.0, sensitivity))  # 限制在0.5-2.0之间
    
    def analyze_voting_patterns(self, thresholds: Dict[str, float]) -> Dict:
        """分析投票模式"""
        if not self.sample_data:
            raise ValueError("没有样本数据")
        
        voting_patterns = defaultdict(int)
        method_contributions = defaultdict(lambda: {'sharp': 0, 'blur': 0})
        
        for sample in self.sample_data:
            votes = []
            for method in self.methods:
                if sample['metrics'][method] > thresholds[method]:
                    votes.append(method)
            
            # 记录投票模式
            vote_count = len(votes)
            voting_patterns[vote_count] += 1
            
            # 记录每个方法的贡献
            is_blur = vote_count <= len(self.methods) / 2
            for method in self.methods:
                if method in votes:
                    if is_blur:
                        method_contributions[method]['blur'] += 1
                    else:
                        method_contributions[method]['sharp'] += 1
        
        # 转换为百分比
        total_samples = len(self.sample_data)
        voting_distribution = {
            f"{k}_votes": v / total_samples * 100 
            for k, v in voting_patterns.items()
        }
        
        return {
            'voting_distribution': voting_distribution,
            'method_contributions': dict(method_contributions),
            'total_samples': total_samples
        }
    
    def visualize_optimization_process(self, optimization_history: List[Dict]):
        """可视化优化过程"""
        iterations = [h['iteration'] for h in optimization_history]
        blur_rates = [h['blur_rate'] for h in optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, blur_rates, 'b-', linewidth=2, label='实际模糊率')
        plt.axhline(y=self.target_blur_rate, color='r', linestyle='--', 
                   label=f'目标模糊率 ({self.target_blur_rate:.1%})')
        
        plt.xlabel('迭代次数')
        plt.ylabel('模糊率')
        plt.title('阈值优化过程')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_optimized_config(self, optimized_thresholds: Dict[str, float], 
                                output_path: Path, analysis_results: Dict = None):
        """生成优化后的配置文件"""
        config = {
            "source_root": "./input",
            "output_root": "./output",
            "classifier_params": {
                f"{method}_threshold": threshold 
                for method, threshold in optimized_thresholds.items()
            },
            "processing": {
                "max_workers": 8,
                "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
            },
            "optimization_info": {
                "target_blur_rate": self.target_blur_rate,
                "achieved_blur_rate": self.simulate_classification(optimized_thresholds),
                "sample_size": len(self.sample_data) if self.sample_data else 0,
                "optimization_method": "smart_voting_aware"
            }
        }
        
        if analysis_results:
            config['voting_analysis'] = analysis_results
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化配置已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="智能阈值优化器 - 考虑投票机制")
    parser.add_argument('folder', help='样本图像文件夹路径')
    parser.add_argument('--target-blur-rate', type=float, default=0.1, 
                       help='目标模糊率 (默认0.1表示10%)')
    parser.add_argument('--sample-size', type=int, default=200, 
                       help='分析的样本数量')
    parser.add_argument('--method', choices=['smart', 'grid'], default='smart',
                       help='优化方法')
    parser.add_argument('--output-config', help='输出配置文件路径')
    parser.add_argument('--analyze-voting', action='store_true', 
                       help='分析投票模式')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = SmartThresholdOptimizer(target_blur_rate=args.target_blur_rate)
    folder_path = Path(args.folder)
    
    if not folder_path.exists():
        logger.error(f"文件夹不存在: {folder_path}")
        return 1
    
    # 收集样本数据
    sample_data = optimizer.collect_sample_data(folder_path, args.sample_size)
    
    if not sample_data:
        logger.error("无法收集到样本数据")
        return 1
    
    # 计算初始统计信息
    logger.info("计算初始统计信息...")
    initial_stats = {}
    for method in optimizer.methods:
        values = [s['metrics'][method] for s in sample_data]
        values = np.array(values)
        
        initial_stats[method] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'percentiles': {}
        }
        
        for p in range(10, 100, 5):
            initial_stats[method]['percentiles'][str(p)] = float(np.percentile(values, p))
    
    # 打印初始统计
    print("\n" + "="*60)
    print("初始统计信息")
    print("="*60)
    for method, stats in initial_stats.items():
        print(f"\n{method}:")
        print(f"  均值: {stats['mean']:.2f} (±{stats['std']:.2f})")
        print(f"  P50: {stats['percentiles']['50']:.2f}")
        print(f"  P70: {stats['percentiles']['70']:.2f}")
        print(f"  P90: {stats['percentiles']['90']:.2f}")
    
    # 使用初始阈值（70百分位）测试
    initial_thresholds = {
        method: stats['percentiles']['70'] 
        for method, stats in initial_stats.items()
    }
    initial_blur_rate = optimizer.simulate_classification(initial_thresholds)
    
    print(f"\n使用70百分位作为阈值的模糊率: {initial_blur_rate:.1%}")
    print(f"目标模糊率: {args.target_blur_rate:.1%}")
    
    # 优化阈值
    print("\n" + "="*60)
    print("开始优化阈值...")
    print("="*60)
    
    if args.method == 'smart':
        optimized_thresholds = optimizer.optimize_thresholds_smart(initial_stats)
    else:
        optimized_thresholds = optimizer.optimize_thresholds_grid_search(initial_stats)
    
    # 验证结果
    final_blur_rate = optimizer.simulate_classification(optimized_thresholds)
    
    print("\n" + "="*60)
    print("优化结果")
    print("="*60)
    print(f"\n最终模糊率: {final_blur_rate:.1%}")
    print(f"目标模糊率: {args.target_blur_rate:.1%}")
    print(f"误差: {abs(final_blur_rate - args.target_blur_rate):.1%}")
    
    print("\n优化后的阈值:")
    for method, threshold in optimized_thresholds.items():
        initial = initial_thresholds[method]
        change = (threshold - initial) / initial * 100
        print(f"  {method}_threshold: {threshold:.2f} "
              f"(初始: {initial:.2f}, 变化: {change:+.1f}%)")
    
    # 分析投票模式
    if args.analyze_voting:
        print("\n" + "="*60)
        print("投票模式分析")
        print("="*60)
        
        analysis = optimizer.analyze_voting_patterns(optimized_thresholds)
        
        print("\n投票分布:")
        for votes, percentage in sorted(analysis['voting_distribution'].items()):
            print(f"  {votes}: {percentage:.1f}%")
        
        print("\n各方法贡献:")
        for method, contrib in analysis['method_contributions'].items():
            total = contrib['sharp'] + contrib['blur']
            if total > 0:
                sharp_ratio = contrib['sharp'] / total * 100
                print(f"  {method}: {sharp_ratio:.1f}% 投票给清晰")
    
    # 保存配置
    if args.output_config:
        analysis_results = None
        if args.analyze_voting:
            analysis_results = optimizer.analyze_voting_patterns(optimized_thresholds)
        
        optimizer.generate_optimized_config(
            optimized_thresholds, 
            Path(args.output_config),
            analysis_results
        )
    
    print("\n" + "="*60)
    print("优化完成！")
    print("="*60)
    
    # 建议的命令行
    print("\n建议使用以下命令运行主程序:")
    print(f"python main.py --config {args.output_config if args.output_config else 'optimized_config.json'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())