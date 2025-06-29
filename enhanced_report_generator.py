import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedReportGenerator:
    """增强的报告生成器"""

    def __init__(self, output_dir: Path):
        """初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_processing_report(self, results: Dict, processing_time: float, 
                                 settings: Dict) -> Dict:
        """生成基础处理报告
        
        Args:
            results: 处理结果
            processing_time: 处理时间
            settings: 处理器设置
            
        Returns:
            报告数据
        """
        report = {
            'processing_summary': {
                'total_time_seconds': round(processing_time, 2),
                'total_folders': len(results.get('processed_folders', [])),
                'total_images': results.get('total_images', 0),
                'clean_images': results.get('clean_images', 0),
                'dirty_images': results.get('dirty_images', 0),
                'same_position_groups': results.get('same_position_groups', 0),
                'error_count': len(results.get('errors', [])),
                'processing_speed': round(results.get('total_images', 0) / processing_time, 2) if processing_time > 0 else 0,
                'clean_ratio': round(results.get('clean_images', 0) / results.get('total_images', 1), 3) if results.get('total_images', 0) > 0 else 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'dirty_reasons_distribution': results.get('dirty_reasons_summary', {}),
            'folder_details': results.get('processed_folders', []),
            'settings': settings,
            'errors': results.get('errors', [])
        }

        # 保存报告
        report_path = self.output_dir / 'enhanced_processing_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"处理报告已保存到: {report_path}")
        return report

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
            
            # 1. 质量分布饼图
            if results.get('total_images', 0) > 0:
                plots_info['quality_pie_chart'] = self._create_quality_pie_chart(results, save_plots)
            
            # 2. 脏数据原因分布图
            if results.get('dirty_reasons_summary'):
                plots_info['dirty_reasons_chart'] = self._create_dirty_reasons_chart(results, save_plots)
            
            # 3. 文件夹处理结果图
            if results.get('processed_folders'):
                plots_info['folder_chart'] = self._create_folder_chart(results, save_plots)
            
            return plots_info
            
        except ImportError:
            logger.warning("matplotlib 未安装，跳过可视化报告生成")
            return {}
        except Exception as e:
            logger.error(f"生成可视化报告时出错: {str(e)}")
            return {}

    def _create_quality_pie_chart(self, results: Dict, save_plot: bool) -> str:
        """创建质量分布饼图"""
        import matplotlib.pyplot as plt
        
        clean = results.get('clean_images', 0)
        dirty = results.get('dirty_images', 0)
        
        if clean + dirty == 0:
            return "no_data"
        
        labels = ['干净数据', '脏数据']
        sizes = [clean, dirty]
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

    def _create_dirty_reasons_chart(self, results: Dict, save_plot: bool) -> str:
        """创建脏数据原因分布图"""
        import matplotlib.pyplot as plt
        
        reasons = results.get('dirty_reasons_summary', {})
        if not reasons:
            return "no_dirty_data"
        
        # 转换原因名称为中文
        reason_names = {
            'blurry': '模糊',
            'overexposed': '过曝',
            'underexposed': '欠曝',
            'out_of_work_area': '离开工作区域',
            'same_position_extra': '同位置重复',
            'analysis_error': '分析错误'
        }
        
        labels = []
        values = []
        for reason, count in reasons.items():
            labels.append(reason_names.get(reason, reason))
            values.append(count)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color='#3498db')
        
        # 在条形图上添加数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.xlabel('原因')
        plt.ylabel('数量')
        plt.title('脏数据原因分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'dirty_reasons_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        return "dirty_reasons_chart_created"

    def _create_folder_chart(self, results: Dict, save_plot: bool) -> str:
        """创建文件夹处理结果图"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        folders = results.get('processed_folders', [])[:10]  # 只显示前10个
        
        if not folders:
            return "no_folder_data"
        
        folder_names = []
        clean_counts = []
        dirty_counts = []
        
        for f in folders:
            folder_names.append(f.get('folder', 'Unknown'))
            clean_counts.append(f.get('clean_images', 0))
            dirty_counts.append(f.get('dirty_images', 0))
        
        x = np.arange(len(folder_names))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, clean_counts, width, label='干净数据', color='#2ecc71')
        plt.bar(x + width/2, dirty_counts, width, label='脏数据', color='#e74c3c')
        
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
        
        return "folder_chart_created"

    def export_to_csv(self, results: Dict) -> str:
        """导出结果到CSV文件
        
        Args:
            results: 处理结果
            
        Returns:
            CSV文件路径
        """
        try:
            import pandas as pd
            
            folders = results.get('processed_folders', [])
            if not folders:
                logger.warning("没有文件夹数据可导出")
                return ""

            # 创建DataFrame
            df_data = []
            for folder in folders:
                clean = folder.get('clean_images', 0)
                dirty = folder.get('dirty_images', 0)
                total = clean + dirty
                
                df_data.append({
                    '文件夹': folder.get('folder', 'Unknown'),
                    '总图像数': total,
                    '干净数据': clean,
                    '脏数据': dirty,
                    '同位置组数': folder.get('same_position_groups', 0),
                    '干净比例': round(clean / total, 3) if total > 0 else 0
                })

            df = pd.DataFrame(df_data)
            csv_path = self.output_dir / 'enhanced_processing_results.csv'
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
        print("增强图像处理摘要")
        print("="*60)
        print(f"处理时间: {summary.get('timestamp', 'N/A')}")
        print(f"总耗时: {summary.get('total_time_seconds', 0)} 秒")
        print(f"处理文件夹数: {summary.get('total_folders', 0)}")
        print(f"总图像数: {summary.get('total_images', 0)}")
        print(f"干净数据: {summary.get('clean_images', 0)} ({summary.get('clean_ratio', 0):.1%})")
        print(f"脏数据: {summary.get('dirty_images', 0)}")
        print(f"同位置拍摄组数: {summary.get('same_position_groups', 0)}")
        
        if report.get('dirty_reasons_distribution'):
            print("\n脏数据原因分布:")
            reason_names = {
                'blurry': '模糊',
                'overexposed': '过曝',
                'underexposed': '欠曝',
                'out_of_work_area': '离开工作区域',
                'same_position_extra': '同位置重复',
                'analysis_error': '分析错误'
            }
            for reason, count in report['dirty_reasons_distribution'].items():
                percentage = count / summary.get('dirty_images', 1) * 100 if summary.get('dirty_images', 0) > 0 else 0
                reason_display = reason_names.get(reason, reason)
                print(f"  - {reason_display}: {count} ({percentage:.1f}%)")
                
        print(f"\n错误数: {summary.get('error_count', 0)}")
        print(f"处理速度: {summary.get('processing_speed', 0)} 图像/秒")
        print("="*60)

    def generate_detailed_analysis(self, results: Dict) -> Dict:
        """生成详细分析报告
        
        Args:
            results: 处理结果
            
        Returns:
            详细分析报告
        """
        analysis = {
            'folder_statistics': self._analyze_folder_statistics(results),
            'quality_metrics': self._analyze_quality_metrics(results),
            'error_analysis': self._analyze_errors(results),
            'performance_metrics': self._calculate_performance_metrics(results)
        }

        # 保存详细分析
        analysis_path = self.output_dir / 'enhanced_detailed_analysis.json'
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
            'folders_with_images': len([f for f in folders if f.get('total_images', 0) > 0]),
            'folders_with_errors': len([f for f in folders if 'error' in f]),
            'average_images_per_folder': sum(f.get('total_images', 0) for f in folders) / len(folders) if folders else 0,
            'folder_quality_rankings': []
        }

        # 计算文件夹质量排名
        for folder in folders:
            total = folder.get('total_images', 0)
            clean = folder.get('clean_images', 0)
            if total > 0:
                stats['folder_quality_rankings'].append({
                    'folder': folder.get('folder', 'Unknown'),
                    'clean_ratio': round(clean / total, 3),
                    'total_images': total,
                    'same_position_groups': folder.get('same_position_groups', 0)
                })

        # 按质量比例排序
        stats['folder_quality_rankings'].sort(key=lambda x: x['clean_ratio'], reverse=True)

        return stats

    def _analyze_quality_metrics(self, results: Dict) -> Dict:
        """分析质量指标"""
        total = results.get('total_images', 0)
        clean = results.get('clean_images', 0)
        dirty = results.get('dirty_images', 0)

        if total == 0:
            return {}

        return {
            'overall_quality_ratio': round(clean / total, 3),
            'dirty_ratio': round(dirty / total, 3),
            'quality_categories': {
                'high_quality_folders': [],  # >80% 干净
                'medium_quality_folders': [],  # 50-80% 干净
                'low_quality_folders': []  # <50% 干净
            }
        }

    def _analyze_errors(self, results: Dict) -> Dict:
        """分析错误信息"""
        errors = results.get('errors', [])
        return {
            'total_errors': len(errors),
            'error_details': errors[:10]  # 只显示前10个错误
        }

    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """计算性能指标"""
        folders = results.get('processed_folders', [])
        if not folders:
            return {}

        folder_sizes = [f.get('total_images', 0) for f in folders if f.get('total_images', 0) > 0]
        
        return {
            'largest_folder_size': max(folder_sizes) if folder_sizes else 0,
            'smallest_folder_size': min(folder_sizes) if folder_sizes else 0,
            'median_folder_size': sorted(folder_sizes)[len(folder_sizes)//2] if folder_sizes else 0,
            'folders_processed': len(folders),
            'success_rate': round(
                len([f for f in folders if 'error' not in f]) / len(folders), 3
            ) if folders else 0
        }