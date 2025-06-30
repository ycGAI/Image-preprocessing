import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedReportGenerator:
    """Enhanced report generator"""

    def __init__(self, output_dir: Path):
        """Initialize report generator
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_processing_report(self, results: Dict, processing_time: float, 
                                 settings: Dict) -> Dict:
        """Generate basic processing report
        
        Args:
            results: Processing results
            processing_time: Processing time
            settings: Processor settings
            
        Returns:
            Report data
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

        # Save report
        report_path = self.output_dir / 'enhanced_processing_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Processing report saved to: {report_path}")
        return report

    def generate_visual_report(self, results: Dict, save_plots: bool = True) -> Dict:
        """Generate visual report
        
        Args:
            results: Processing results
            save_plots: Whether to save plots
            
        Returns:
            Visual report information
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('default')
            
            # Set Chinese font
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            plots_info = {}
            
            # 1. Quality distribution pie chart
            if results.get('total_images', 0) > 0:
                plots_info['quality_pie_chart'] = self._create_quality_pie_chart(results, save_plots)
            
            # 2. Dirty reason distribution chart
            if results.get('dirty_reasons_summary'):
                plots_info['dirty_reasons_chart'] = self._create_dirty_reasons_chart(results, save_plots)
            
            # 3. Folder processing results chart
            if results.get('processed_folders'):
                plots_info['folder_chart'] = self._create_folder_chart(results, save_plots)
            
            return plots_info
            
        except ImportError:
            logger.warning("matplotlib not installed, skipping visual report generation")
            return {}
        except Exception as e:
            logger.error(f"Error generating visual report: {str(e)}")
            return {}

    def _create_quality_pie_chart(self, results: Dict, save_plot: bool) -> str:
        """Create quality distribution pie chart"""
        import matplotlib.pyplot as plt
        
        clean = results.get('clean_images', 0)
        dirty = results.get('dirty_images', 0)
        
        if clean + dirty == 0:
            return "no_data"
        
        labels = ['Clean Data', 'Dirty Data']
        sizes = [clean, dirty]
        colors = ['#2ecc71', '#e74c3c']
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Image Quality Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        if save_plot:
            plot_path = self.output_dir / 'quality_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        return "quality_pie_chart_created"

    def _create_dirty_reasons_chart(self, results: Dict, save_plot: bool) -> str:
        """Create dirty data reason distribution chart"""
        import matplotlib.pyplot as plt
        
        reasons = results.get('dirty_reasons_summary', {})
        if not reasons:
            return "no_dirty_data"

        reason_names = {
            'blurry': 'Blurry',
            'overexposed': 'Overexposed',
            'underexposed': 'Underexposed',
            'out_of_work_area': 'Out of Work Area',
            'same_position_extra': 'Same Position Duplicate',
            'analysis_error': 'Analysis Error'
        }
        
        labels = []
        values = []
        for reason, count in reasons.items():
            labels.append(reason_names.get(reason, reason))
            values.append(count)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color='#3498db')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.xlabel('Reason')
        plt.ylabel('Count')
        plt.title('Dirty Data Reason Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'dirty_reasons_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        return "dirty_reasons_chart_created"

    def _create_folder_chart(self, results: Dict, save_plot: bool) -> str:
        """Create folder processing results chart"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        folders = results.get('processed_folders', [])[:10]  # Show only first 10
        
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
        plt.bar(x - width/2, clean_counts, width, label='Clean Data', color='#2ecc71')
        plt.bar(x + width/2, dirty_counts, width, label='Dirty Data', color='#e74c3c')
        
        plt.xlabel('Folder')
        plt.ylabel('Image Count')
        plt.title('Image Quality Distribution by Folder')
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
        """Export results to CSV file
        
        Args:
            results: Processing results
            
        Returns:
            CSV file path
        """
        try:
            import pandas as pd
            
            folders = results.get('processed_folders', [])
            if not folders:
                logger.warning("No folder data to export")
                return ""

            # Create DataFrame
            df_data = []
            for folder in folders:
                clean = folder.get('clean_images', 0)
                dirty = folder.get('dirty_images', 0)
                total = clean + dirty
                
                df_data.append({
                    'Folder': folder.get('folder', 'Unknown'),
                    'Total Images': total,
                    'Clean Data': clean,
                    'Dirty Data': dirty,
                    'Same Position Groups': folder.get('same_position_groups', 0),
                    'Clean Ratio': round(clean / total, 3) if total > 0 else 0
                })

            df = pd.DataFrame(df_data)
            csv_path = self.output_dir / 'enhanced_processing_results.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"Results exported to CSV: {csv_path}")
            return str(csv_path)
            
        except ImportError:
            logger.warning("pandas not installed, cannot export CSV")
            return ""
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            return ""

    def print_summary(self, report: Dict):
        """Print processing summary
        
        Args:
            report: Report data
        """
        summary = report.get('processing_summary', {})
        print("\n" + "="*60)
        print("Enhanced Image Processing Summary")
        print("="*60)
        print(f"Processing Time: {summary.get('timestamp', 'N/A')}")
        print(f"Total Time: {summary.get('total_time_seconds', 0)} seconds")
        print(f"Folders Processed: {summary.get('total_folders', 0)}")
        print(f"Total Images: {summary.get('total_images', 0)}")
        print(f"Clean Data: {summary.get('clean_images', 0)} ({summary.get('clean_ratio', 0):.1%})")
        print(f"Dirty Data: {summary.get('dirty_images', 0)}")
        print(f"Same Position Groups: {summary.get('same_position_groups', 0)}")
        
        if report.get('dirty_reasons_distribution'):
            print("\nDirty Data Reason Distribution:")
            reason_names = {
                'blurry': 'Blurry',
                'overexposed': 'Overexposed',
                'underexposed': 'Underexposed',
                'out_of_work_area': 'Out of Work Area',
                'same_position_extra': 'Same Position Duplicate',
                'analysis_error': 'Analysis Error'
            }
            for reason, count in report['dirty_reasons_distribution'].items():
                percentage = count / summary.get('dirty_images', 1) * 100 if summary.get('dirty_images', 0) > 0 else 0
                reason_display = reason_names.get(reason, reason)
                print(f"  - {reason_display}: {count} ({percentage:.1f}%)")
                
        print(f"\nErrors: {summary.get('error_count', 0)}")
        print(f"Processing Speed: {summary.get('processing_speed', 0)} images/second")
        print("="*60)

    def generate_detailed_analysis(self, results: Dict) -> Dict:
        """Generate detailed analysis report
        
        Args:
            results: Processing results
            
        Returns:
            Detailed analysis report
        """
        analysis = {
            'folder_statistics': self._analyze_folder_statistics(results),
            'quality_metrics': self._analyze_quality_metrics(results),
            'error_analysis': self._analyze_errors(results),
            'performance_metrics': self._calculate_performance_metrics(results)
        }

        # Save detailed analysis
        analysis_path = self.output_dir / 'enhanced_detailed_analysis.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        logger.info(f"Detailed analysis report saved to: {analysis_path}")
        return analysis

    def _analyze_folder_statistics(self, results: Dict) -> Dict:
        """Analyze folder statistics"""
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

        # Calculate folder quality rankings
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

        # Sort by quality ratio
        stats['folder_quality_rankings'].sort(key=lambda x: x['clean_ratio'], reverse=True)

        return stats

    def _analyze_quality_metrics(self, results: Dict) -> Dict:
        """Analyze quality metrics"""
        total = results.get('total_images', 0)
        clean = results.get('clean_images', 0)
        dirty = results.get('dirty_images', 0)

        if total == 0:
            return {}

        return {
            'overall_quality_ratio': round(clean / total, 3),
            'dirty_ratio': round(dirty / total, 3),
            'quality_categories': {
                'high_quality_folders': [],  # >80% clean
                'medium_quality_folders': [],  # 50-80% clean
                'low_quality_folders': []  # <50% clean
            }
        }

    def _analyze_errors(self, results: Dict) -> Dict:
        """Analyze error information"""
        errors = results.get('errors', [])
        return {
            'total_errors': len(errors),
            'error_details': errors[:10]  # Show only first 10 errors
        }

    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics"""
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