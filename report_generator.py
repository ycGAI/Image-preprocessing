import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: Path):
        """初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_processing_report(self, results: Dict, processing_time: float, 
                                 classifier_settings: Dict) -> Dict:
        """生成基础处理报告
        
        Args:
            results: 处理结果
            processing_time: 处理时间
            classifier_settings: 分类器设置
            
        Returns:
            报告数据
        """
        report = {
            'processing_summary': {
                'total_time_seconds': round(processing_time, 2),
                'total_folders': len(results.get('processed_folders', [])),
                'total_images': results.get('total_images', 0),
                'sharp_images': results.get('sharp_images', 0),
                'blurry_images': results.get('blurry_images', 0),
                'error_count': len(results.get('errors', [])),
                'processing_speed': round(results.get('total_images', 0) / processing_time, 2) if processing_time > 0 else 0,
                'sharp_ratio': round(results.get('sharp_images', 0) / results.get('total_images', 1), 3) if results.get('total_images', 0) > 0 else 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'folder_details': results.get('processed_folders', []),
            'classifier_settings': classifier_settings,
            'errors': results.get('errors', [])
        }

        # 保存报告
        report_path = self.output_dir / 'processing_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"处理报告已保存到: {report_path}")
        return report

    def generate_detailed_analysis(self, results: Dict) -> Dict:
        """生成详细分析报告
        
        Args:
            results: 处理结果
            
        Returns:
            详细分析报告
        """
        analysis = {
            'folder_statistics': self._analyze_folder_statistics(results),
            'quality_distribution': self._analyze_quality_distribution(results),
            'error_analysis': self._analyze_errors(results),
            'performance_metrics': self._calculate_performance_metrics(results)
        }

        # 保存详细分析
        analysis_path = self.output_dir / 'detailed_analysis.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        logger.info(f"详细分析报告已保存到: {analysis_path}")
        return analysis

    def _analyze_folder_statistics(self, results: Dict) -> Dict:
        """分析文件夹统计信息"""
        folders = results.get('processed_folders', [])
        if not folders:
            return {}

        stats = {
            'total_folders': len(folders),
            'folders_with_images': len([f for f in folders if f['processed'] > 0]),
            'folders_with_errors': len([f for f in folders if f['errors'] > 0]),
            'average_images_per_folder': sum(f['processed'] for f in folders) / len(folders),
            'folder_quality_ratios': []
        }

        for folder in folders:
            if folder['processed'] > 0:
                sharp_ratio = folder['sharp'] / folder['processed']
                stats['folder_quality_ratios'].append({
                    'folder': folder['folder'],
                    'sharp_ratio': round(sharp_ratio, 3),
                    'total_images': folder['processed']
                })

        # 排序文件夹质量比例
        stats['folder_quality_ratios'].sort(key=lambda x: x['sharp_ratio'], reverse=True)

        return stats

    def _analyze_quality_distribution(self, results: Dict) -> Dict:
        """分析质量分布"""
        total_images = results.get('total_images', 0)
        sharp_images = results.get('sharp_images', 0)
        blurry_images = results.get('blurry_images', 0)

        if total_images == 0:
            return {}

        return {
            'sharp_percentage': round(sharp_images / total_images * 100, 2),
            'blurry_percentage': round(blurry_images / total_images * 100, 2),
            'quality_categories': {
                'high_quality_folders': [],  # >80% 清晰
                'medium_quality_folders': [],  # 50-80% 清晰
                'low_quality_folders': []  # <50% 清晰
            }
        }

    def _analyze_errors(self, results: Dict) -> Dict:
        """分析错误信息"""
        errors = results.get('errors', [])
        if not errors:
            return {'total_errors': 0, 'error_types': {}}

        error_types = defaultdict(int)
        for error in errors:
            # 简单的错误分类
            if '无法读取图像' in error:
                error_types['image_read_error'] += 1
            elif '文件夹' in error and '出错' in error:
                error_types['folder_processing_error'] += 1
            else:
                error_types['other_error'] += 1

        return {
            'total_errors': len(errors),
            'error_types': dict(error_types),
            'error_details': errors
        }

    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """计算性能指标"""
        folders = results.get('processed_folders', [])
        if not folders:
            return {}

        folder_sizes = [f['processed'] for f in folders if f['processed'] > 0]
        
        return {
            'largest_folder_size': max(folder_sizes) if folder_sizes else 0,
            'smallest_folder_size': min(folder_sizes) if folder_sizes else 0,
            'median_folder_size': sorted(folder_sizes)[len(folder_sizes)//2] if folder_sizes else 0,
            'folders_processed': len(folders),
            'success_rate': round(len([f for f in folders if f['errors'] == 0]) / len(folders), 3) if folders else 0
        }

    def generate_visual_report(self, results: Dict, save_plots: bool = True) -> Dict:
        """生成可视化报告
        
        Args:
            results: 处理结果
            save_plots: 是否保存图表
            
        Returns:
            可视化报告信息
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('default')
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            plots_info = {}
            
            # 1. 整体质量分布饼图
            if results.get('total_images', 0) > 0:
                plots_info['quality_pie_chart'] = self._create_quality_pie_chart(results, save_plots)
            
            # 2. 文件夹处理结果柱状图
            if results.get('processed_folders'):
                plots_info['folder_bar_chart'] = self._create_folder_bar_chart(results, save_plots)
            
            # 3. 清晰度比例分布直方图
            if results.get('processed_folders'):
                plots_info['quality_histogram'] = self._create_quality_histogram(results, save_plots)
            
            return plots_info
            
        except ImportError:
            logger.warning("matplotlib 未安装，跳过可视化报告生成")
            return {}
        except Exception as e:
            logger.error(f"生成可视化报告时出错: {str(e)}")
            return {}

    def _create_quality_pie_chart(self, results: Dict, save_plot: bool) -> str:
        """创建质量分布饼图"""
        sharp = results.get('sharp_images', 0)
        blurry = results.get('blurry_images', 0)
        
        labels = ['sharp', 'blurry']
        sizes = [sharp, blurry]
        colors = ['#2ecc71', '#e74c3c']
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('图像质量分布', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        if save_plot:
            plot_path = self.output_dir / 'quality_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        return "quality_pie_chart_created"

    def _create_folder_bar_chart(self, results: Dict, save_plot: bool) -> str:
        """创建文件夹处理结果柱状图"""
        folders = results.get('processed_folders', [])[:10]  # 只显示前10个文件夹
        
        folder_names = [f['folder'] for f in folders]
        sharp_counts = [f['sharp'] for f in folders]
        blurry_counts = [f['blurry'] for f in folders]
        
        x = range(len(folder_names))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar([i - width/2 for i in x], sharp_counts, width, label='sharp', color='#2ecc71')
        plt.bar([i + width/2 for i in x], blurry_counts, width, label='blurry', color='#e74c3c')
        
        plt.xlabel('文件夹')
        plt.ylabel('图像数量')
        plt.title('各文件夹图像质量分布')
        plt.xticks(x, folder_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'folder_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        return "folder_bar_chart_created"

    def _create_quality_histogram(self, results: Dict, save_plot: bool) -> str:
        """创建清晰度比例分布直方图"""
        folders = results.get('processed_folders', [])
        ratios = []
        
        for folder in folders:
            if folder['processed'] > 0:
                ratio = folder['sharp'] / folder['processed']
                ratios.append(ratio)
        
        if not ratios:
            return "no_data_for_histogram"
        
        plt.figure(figsize=(10, 6))
        plt.hist(ratios, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        plt.xlabel('清晰图像比例')
        plt.ylabel('文件夹数量')
        plt.title('清晰度比例分布直方图')
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plot_path = self.output_dir / 'quality_ratio_histogram.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        return "quality_histogram_created"

    def export_to_csv(self, results: Dict) -> str:
        """导出结果到CSV文件
        
        Args:
            results: 处理结果
            
        Returns:
            CSV文件路径
        """
        try:
            folders = results.get('processed_folders', [])
            if not folders:
                logger.warning("没有文件夹数据可导出")
                return ""

            # 创建DataFrame
            df_data = []
            for folder in folders:
                df_data.append({
                    '文件夹': folder['folder'],
                    '总图像数': folder['processed'],
                    '清晰图像数': folder['sharp'],
                    '模糊图像数': folder['blurry'],
                    '错误数': folder['errors'],
                    '清晰比例': round(folder['sharp'] / folder['processed'], 3) if folder['processed'] > 0 else 0
                })

            df = pd.DataFrame(df_data)
            csv_path = self.output_dir / 'processing_results.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"结果已导出到CSV: {csv_path}")
            return str(csv_path)
            
        except ImportError:
            logger.warning("pandas 未安装，无法导出CSV")
            return ""
        except Exception as e:
            logger.error(f"导出CSV时出错: {str(e)}")
            return ""

    def print_summary(self, report: Dict):
        """打印处理摘要
        
        Args:
            report: 报告数据
        """
        summary = report.get('processing_summary', {})
        print("\n" + "="*60)
        print("图像清晰度分类处理摘要")
        print("="*60)
        print(f"处理时间: {summary.get('timestamp', 'N/A')}")
        print(f"总耗时: {summary.get('total_time_seconds', 0)} 秒")
        print(f"处理文件夹数: {summary.get('total_folders', 0)}")
        print(f"总图像数: {summary.get('total_images', 0)}")
        print(f"清晰图像: {summary.get('sharp_images', 0)} ({summary.get('sharp_ratio', 0):.1%})")
        print(f"模糊图像: {summary.get('blurry_images', 0)}")
        print(f"错误数: {summary.get('error_count', 0)}")
        print(f"处理速度: {summary.get('processing_speed', 0)} 图像/秒")
        print("="*60)