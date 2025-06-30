#!/usr/bin/env python3
"""
Fixed Working Threshold Analyzer - Generates compatible format config files

Main fixes:
1. Generates config files fully compatible with original format
2. Retains all required fields
3. Supports custom paths
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
    """Batch image analysis worker function"""
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
            logger.error(f"Failed to analyze image {image_path}: {e}")
            results.append(None)
    
    return results


class FixedWorkingAnalyzer:
    """Fixed working threshold analyzer"""
    
    def __init__(self, target_blur_rate: float = 0.1, n_workers: int = None):
        self.target_blur_rate = target_blur_rate
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        self.methods = ['laplacian', 'sobel', 'brenner', 'tenengrad']
        self.sample_data = None
        self.train_data = None
        self.val_data = None
        self.n_workers = n_workers or cpu_count()
        
        # Config file template
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
        """Collect and analyze sample data"""
        start_time = time.time()
        logger.info(f"Collecting sample data from {folder_path}...")
        logger.info(f"Using {self.n_workers} worker processes")
        
        # Collect image files
        image_files = []
        for fmt in self.supported_formats:
            image_files.extend(folder_path.glob(f'**/*{fmt}'))
            image_files.extend(folder_path.glob(f'**/*{fmt.upper()}'))
        
        if len(image_files) > sample_size:
            image_files = random.sample(image_files, sample_size)
        
        logger.info(f"Analyzing {len(image_files)} image files...")
        
        # Process in batches
        batches = []
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            batches.append((batch, max_image_size))
        
        # Multi-process handling
        sample_data = []
        failed_count = 0
        
        with Pool(self.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(analyze_image_batch, batches),
                total=len(batches),
                desc="Processing batches"
            ))
        
        for batch_results in results:
            for result in batch_results:
                if result:
                    sample_data.append(result)
                else:
                    failed_count += 1
        
        self.sample_data = sample_data
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully analyzed {len(sample_data)} images, failed {failed_count}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds ({len(sample_data)/elapsed_time:.1f} images/second)")
        
        # Analyze data distribution
        distribution_analysis = self._analyze_distribution(sample_data)
        
        return distribution_analysis
    
    def _analyze_distribution(self, sample_data: List[Dict]) -> Dict:
        """Analyze data distribution"""
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
        """Safe visualization (save only, no display)"""
        if not self.sample_data:
            logger.warning("No sample data to visualize")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            for idx, method in enumerate(self.methods):
                values = [s['metrics'][method] for s in self.sample_data]
                
                ax = axes[idx]
                ax.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Add percentile lines
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
                logger.info(f"Distribution plot saved to: {output_file}")
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            logger.info("Continuing with other steps...")
    
    def split_data(self, test_size: float = 0.3):
        """Split data"""
        if not self.sample_data:
            raise ValueError("No sample data")
        
        indices = np.arange(len(self.sample_data))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        self.train_data = [self.sample_data[i] for i in train_idx]
        self.val_data = [self.sample_data[i] for i in val_idx]
        
        logger.info(f"Data split complete: training set {len(self.train_data)}, validation set {len(self.val_data)}")
    
    def _simulate_classification(self, thresholds: Dict[str, float], 
                               data: Optional[List[Dict]] = None) -> float:
        """Simulate classification process"""
        if data is None:
            data = self.sample_data
        
        if not data:
            raise ValueError("No data for simulation")
        
        blur_count = 0
        for sample in data:
            votes = sum(1 for m in self.methods if sample['metrics'][m] > thresholds[m])
            if votes <= len(self.methods) / 2:
                blur_count += 1
        
        return blur_count / len(data)
    
    def optimize_thresholds_simple(self, data: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Simple but effective threshold optimization - fixed version"""
        if data is None:
            data = self.train_data if self.train_data else self.sample_data
        
        if not data:
            raise ValueError("No data for optimization")
        
        logger.info("Starting threshold optimization...")
        
        # Collect values for each method
        method_values = {}
        for method in self.methods:
            method_values[method] = np.array([d['metrics'][method] for d in data])
        
        # Fix: If target is 10% blur rate, it means 90% should be judged as sharp
        # So initial threshold should be set at lower percentile
        # Use 10th percentile as initial value (90% of samples will be above this threshold)
        target_clear_rate = 1 - self.target_blur_rate  # 0.9
        initial_percentile = (1 - target_clear_rate) * 100  # 10
        
        current_thresholds = {}
        for method in self.methods:
            current_thresholds[method] = np.percentile(method_values[method], initial_percentile)
        
        logger.info(f"Initial thresholds set at {initial_percentile:.0f}th percentile")
        
        # Iterative optimization
        best_thresholds = current_thresholds.copy()
        best_error = float('inf')
        
        # Fix: Search range should be in lower percentile range
        # For 10% blur rate, search range approximately 5-30 percentile
        percentile_range = range(5, 31, 1)
        
        logger.info("Searching for best percentile combination...")
        for base_percentile in tqdm(percentile_range, desc="Optimization progress"):
            # Try using same percentile for all methods
            test_thresholds = {}
            for method in self.methods:
                test_thresholds[method] = np.percentile(method_values[method], base_percentile)
            
            blur_rate = self._simulate_classification(test_thresholds, data)
            error = abs(blur_rate - self.target_blur_rate)
            
            if error < best_error:
                best_error = error
                best_thresholds = test_thresholds.copy()
                
                if error < 0.01:  # Error less than 1%
                    logger.info(f"Found satisfactory thresholds (error {error:.3f}, percentile {base_percentile})")
                    break
        
        # Fine-tune each method
        logger.info("Fine-tuning thresholds for each method...")
        for method in self.methods:
            current_value = best_thresholds[method]
            best_value = current_value
            local_best_error = best_error
            
            # Search within ±20% of current value
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
        
        # Verify final result
        final_blur_rate = self._simulate_classification(best_thresholds, data)
        logger.info(f"Optimization complete, minimum error: {best_error:.4f}")
        logger.info(f"Final blur rate: {final_blur_rate:.1%} (target: {self.target_blur_rate:.1%})")
        
        return best_thresholds
    
    def validate_thresholds(self, thresholds: Dict[str, float]) -> Dict:
        """Validate thresholds"""
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
        """Generate report and compatible format config file"""
        
        # Generate compatible format config file
        config_path = output_path / 'optimized_config.json'
        
        # Create complete config
        config = {
            "source_root": source_root or self.config_template["source_root"],
            "output_root": output_root or self.config_template["output_root"],
            "classifier_params": {
                "laplacian_threshold": float(optimized_thresholds.get('laplacian', 100.0)),
                "sobel_threshold": float(optimized_thresholds.get('sobel', 50.0)),
                "brenner_threshold": float(optimized_thresholds.get('brenner', 1000.0)),
                "tenengrad_threshold": float(optimized_thresholds.get('tenengrad', 500.0)),
                "variance_threshold": 50.0  # Keep original variance_threshold
            },
            "processing": self.config_template["processing"],
            "logging": self.config_template["logging"],
            "reports": self.config_template["reports"]
        }
        
        # Add optimization info (as comment or extra field)
        config["_optimization_info"] = {
            "target_blur_rate": self.target_blur_rate,
            "achieved_blur_rate": validation_results.get('validation_blur_rate', 0),
            "validation_error": validation_results.get('error', 0),
            "sample_size": len(self.sample_data) if self.sample_data else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimized config saved to: {config_path}")
        
        # Generate detailed report
        report_path = output_path / 'optimization_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Threshold Optimization Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sample count: {len(self.sample_data)}\n")
            f.write(f"Target blur rate: {self.target_blur_rate:.1%}\n")
            f.write(f"Achieved blur rate: {validation_results.get('validation_blur_rate', 0):.1%}\n")
            f.write(f"Error: {validation_results.get('error', 0):.1%}\n\n")
            
            f.write("Optimized thresholds:\n")
            for method, threshold in optimized_thresholds.items():
                if method in distribution_analysis['methods']:
                    stats = distribution_analysis['methods'][method]
                    # Find closest percentile
                    percentile = "Unknown"
                    for p in sorted(stats['percentiles'].keys()):
                        if threshold <= stats['percentiles'][p]:
                            percentile = f"~{p}th percentile"
                            break
                    else:
                        percentile = ">99th percentile"
                    
                    f.write(f"  {method}_threshold: {threshold:.2f} ({percentile})\n")
            
            f.write("\nData distribution statistics:\n")
            for method, stats in distribution_analysis['methods'].items():
                f.write(f"\n{method}:\n")
                f.write(f"  Mean: {stats['mean']:.2f}\n")
                f.write(f"  Std Dev: {stats['std']:.2f}\n")
                f.write(f"  Median: {stats['median']:.2f}\n")
                f.write(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n")
        
        logger.info(f"Optimization report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Fixed Working Threshold Analyzer")
    
    parser.add_argument('folder', help='Sample image folder path')
    parser.add_argument('--target-blur-rate', type=float, default=0.1,
                       help='Target blur rate (default 0.1 means 10%%)')
    parser.add_argument('--sample-size', type=int, default=500,
                       help='Number of samples to analyze')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes')
    parser.add_argument('--validate', action='store_true',
                       help='Validate results using validation set')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate and save visualization charts')
    parser.add_argument('--source-root', type=str, 
                       default="/media/gyc/Backup Plus5/gyc/ATB_data/raw_data",
                       help='Source root directory path')
    parser.add_argument('--output-root', type=str,
                       default="/media/gyc/Backup Plus5/gyc/ATB_data/output_test",
                       help='Output root directory path')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create analyzer
    analyzer = FixedWorkingAnalyzer(
        target_blur_rate=args.target_blur_rate,
        n_workers=args.workers
    )
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return 1
    
    # Record total time
    total_start = time.time()
    
    # Step 1: Collect and analyze data
    print("\n" + "="*60)
    print("Step 1: Data Collection and Distribution Analysis")
    print("="*60)
    
    distribution_analysis = analyzer.collect_and_analyze_samples(
        folder_path, 
        sample_size=args.sample_size
    )
    
    # Print distribution summary
    print("\nData distribution summary:")
    for method, stats in distribution_analysis['methods'].items():
        print(f"\n{method}:")
        print(f"  Mean: {stats['mean']:.2f} (±{stats['std']:.2f})")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  P70: {stats['percentiles'][70]:.2f}")
    
    # Visualization (safe mode)
    if args.visualize:
        print("\nGenerating visualization charts...")
        analyzer.visualize_distribution_safe(save_path=args.output_dir)
    
    # Step 2: Data split
    if args.validate:
        print("\n" + "="*60)
        print("Step 2: Data Split")
        print("="*60)
        analyzer.split_data(test_size=0.3)
    
    # Step 3: Threshold optimization
    print("\n" + "="*60)
    print("Step 3: Threshold Optimization")
    print("="*60)
    
    # Initial thresholds
    initial_thresholds = {}
    for method, stats in distribution_analysis['methods'].items():
        initial_thresholds[method] = stats['percentiles'][70]
    
    initial_blur_rate = analyzer._simulate_classification(initial_thresholds)
    print(f"\nInitial blur rate (P70): {initial_blur_rate:.1%}")
    print(f"Target blur rate: {args.target_blur_rate:.1%}")
    
    # Optimization
    opt_start = time.time()
    optimized_thresholds = analyzer.optimize_thresholds_simple()
    opt_time = time.time() - opt_start
    
    # Display optimization results
    optimized_blur_rate = analyzer._simulate_classification(optimized_thresholds)
    print(f"\nOptimization complete (took {opt_time:.1f} seconds)")
    print(f"Optimized blur rate: {optimized_blur_rate:.1%}")
    print("\nOptimized thresholds:")
    for method, threshold in optimized_thresholds.items():
        initial = initial_thresholds[method]
        change = (threshold - initial) / initial * 100
        print(f"  {method}: {threshold:.2f} (change: {change:+.1f}%)")
    
    # Step 4: Validation
    validation_results = {}
    if args.validate:
        print("\n" + "="*60)
        print("Step 4: Validation Set Validation")
        print("="*60)
        
        validation_results = analyzer.validate_thresholds(optimized_thresholds)
        print(f"\nValidation set blur rate: {validation_results['validation_blur_rate']:.1%}")
        print(f"Validation error: {validation_results['error']:.1%}")
    else:
        validation_results = {
            'validation_blur_rate': optimized_blur_rate,
            'error': abs(optimized_blur_rate - args.target_blur_rate)
        }
    
    # Step 5: Generate report
    print("\n" + "="*60)
    print("Step 5: Generate Report")
    print("="*60)
    
    analyzer.generate_report(
        optimized_thresholds,
        distribution_analysis,
        validation_results,
        args.output_dir,
        source_root=args.source_root,
        output_root=args.output_root
    )
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print(f"Optimization complete! Total time: {total_time:.1f} seconds")
    print(f"Config file saved to: {args.output_dir / 'optimized_config.json'}")
    print("\nSuggested command to run main program:")
    print(f"python main.py --config {args.output_dir / 'optimized_config.json'}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())