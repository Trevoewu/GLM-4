 #!/usr/bin/env python3
"""
Comprehensive visualization utility for CMCC-34 data augmentation analysis.
Consolidates all plotting functionality into a single, easy-to-use module.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Set up plotting style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class DataVisualizer:
    """Comprehensive data visualization for CMCC-34 augmentation analysis."""
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Business type mapping for readable labels
        self.business_types = {
            0: "咨询（含查询）业务规定", 1: "办理取消", 2: "咨询（含查询）业务资费",
            3: "咨询（含查询）营销活动信息", 4: "咨询（含查询）办理方式", 
            5: "投诉（含抱怨）业务使用问题", 6: "咨询（含查询）账户信息",
            7: "办理开通", 8: "咨询（含查询）业务订购信息查询",
            9: "投诉（含抱怨）不知情定制问题", 10: "咨询（含查询）产品/业务功能",
            11: "咨询（含查询）用户资料", 12: "投诉（含抱怨）费用问题",
            13: "投诉（含抱怨）业务办理问题", 14: "投诉（含抱怨）服务问题",
            15: "办理变更", 16: "咨询（含查询）服务渠道信息",
            17: "投诉（含抱怨）业务规定不满", 18: "投诉（含抱怨）营销问题",
            19: "投诉（含抱怨）网络问题", 20: "办理停复机",
            21: "投诉（含抱怨）信息安全问题", 22: "办理重置/修改/补发",
            23: "咨询（含查询）使用方式", 24: "咨询（含查询）号码状态",
            25: "咨询（含查询）工单处理结果", 26: "办理打印/邮寄",
            27: "咨询（含查询）宽带覆盖范围", 28: "办理移机/装机/拆机",
            29: "办理缴费", 30: "办理下载/设置", 31: "办理补换卡",
            32: "办理销户/重开", 33: "咨询（含查询）电商货品信息"
        }
    
    def load_dataset(self, file_path: str) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Load dataset and return DataFrame with class distribution analysis."""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None, {}, {}
        
        print(f"📊 Loading dataset: {file_path}")
        df = pd.read_csv(file_path)
        class_counts = Counter(df['c_numerical'])
        
        # Calculate imbalance metrics
        counts = list(class_counts.values())
        metrics = {
            'total_samples': len(df),
            'num_classes': len(class_counts),
            'min_samples': min(counts),
            'max_samples': max(counts),
            'mean_samples': np.mean(counts),
            'std_samples': np.std(counts),
            'imbalance_ratio': max(counts) / min(counts),
            'gini_coefficient': self._calculate_gini(counts),
            'minority_classes': len([c for c in counts if c < 50]),
            'ultra_minority_classes': len([c for c in counts if c < 10])
        }
        
        print(f"   Total samples: {metrics['total_samples']:,}")
        print(f"   Classes: {metrics['num_classes']}")
        print(f"   Imbalance ratio: {metrics['imbalance_ratio']:.1f}:1")
        print(f"   Gini coefficient: {metrics['gini_coefficient']:.3f}")
        
        return df, class_counts, metrics
    
    def _calculate_gini(self, counts: List[int]) -> float:
        """Calculate Gini coefficient for class distribution."""
        if not counts:
            return 0.0
        
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def plot_class_distribution(self, class_counts: Dict, title: str, 
                              save_name: Optional[str] = None) -> None:
        """Create comprehensive class distribution visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Class Distribution Analysis: {title}', fontsize=16, fontweight='bold')
        
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        
        # Plot 1: Bar chart
        ax1 = axes[0, 0]
        bars = ax1.bar(classes, counts, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_title('Sample Count by Class')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Sample Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Log scale bar chart
        ax2 = axes[0, 1]
        log_counts = [np.log10(c + 1) for c in counts]
        ax2.bar(classes, log_counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax2.set_title('Sample Count (Log Scale)')
        ax2.set_xlabel('Class ID')
        ax2.set_ylabel('Sample Count (log10)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Top 10 classes pie chart
        ax3 = axes[1, 0]
        top_10 = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_counts = [item[1] for item in top_10]
        top_10_labels = [f'Class {item[0]}' for item in top_10]
        
        ax3.pie(top_10_counts, labels=top_10_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Top 10 Classes Distribution')
        
        # Plot 4: Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Dataset Statistics:
        
        Total Samples: {sum(counts):,}
        Number of Classes: {len(classes)}
        Min Samples: {min(counts):,}
        Max Samples: {max(counts):,}
        Mean Samples: {np.mean(counts):.1f}
        Std Deviation: {np.std(counts):.1f}
        Imbalance Ratio: {max(counts)/min(counts):.1f}:1
        Gini Coefficient: {self._calculate_gini(counts):.3f}
        Minority Classes (<50): {len([c for c in counts if c < 50])}
        Ultra Minority (<10): {len([c for c in counts if c < 10])}
        """
        
        ax4.text(0.1, 0.9, stats_text, fontsize=11, transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        plt.show()
    
    def plot_before_after_comparison(self, original_file: str, balanced_file: str,
                                   save_name: str = "before_after_comparison") -> None:
        """Create comprehensive before/after comparison visualization."""
        print("\n" + "="*60)
        print("🔄 BEFORE/AFTER AUGMENTATION COMPARISON")
        print("="*60)
        
        # Load datasets
        original_df, original_counts, original_metrics = self.load_dataset(original_file)
        balanced_df, balanced_counts, balanced_metrics = self.load_dataset(balanced_file)
        
        if original_df is None or balanced_df is None:
            print("❌ Cannot create comparison - one or both datasets failed to load")
            return
        
        # Create comprehensive comparison
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('CMCC-34 Dataset: Before vs After Augmentation', 
                    fontsize=18, fontweight='bold')
        
        # Get all classes
        all_classes = sorted(set(original_counts.keys()) | set(balanced_counts.keys()))
        
        # Plot 1: Side-by-side comparison
        ax1 = plt.subplot(3, 3, 1)
        width = 0.35
        x = np.arange(len(all_classes))
        
        original_values = [original_counts.get(c, 0) for c in all_classes]
        balanced_values = [balanced_counts.get(c, 0) for c in all_classes]
        
        ax1.bar(x - width/2, original_values, width, label='Original', 
               alpha=0.7, color='lightcoral')
        ax1.bar(x + width/2, balanced_values, width, label='Balanced', 
               alpha=0.7, color='lightblue')
        
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Sample Count')
        ax1.set_title('Sample Count Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_classes, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log scale comparison
        ax2 = plt.subplot(3, 3, 2)
        ax2.bar(x - width/2, original_values, width, label='Original', 
               alpha=0.7, color='lightcoral')
        ax2.bar(x + width/2, balanced_values, width, label='Balanced', 
               alpha=0.7, color='lightblue')
        ax2.set_yscale('log')
        ax2.set_xlabel('Class ID')
        ax2.set_ylabel('Sample Count (log scale)')
        ax2.set_title('Sample Count Comparison (Log Scale)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_classes, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Minority classes focus
        ax3 = plt.subplot(3, 3, 3)
        minority_classes = [c for c in all_classes if original_counts.get(c, 0) < 50]
        minority_original = [original_counts.get(c, 0) for c in minority_classes]
        minority_balanced = [balanced_counts.get(c, 0) for c in minority_classes]
        
        x_minority = np.arange(len(minority_classes))
        ax3.bar(x_minority - width/2, minority_original, width, label='Original', 
               alpha=0.7, color='lightcoral')
        ax3.bar(x_minority + width/2, minority_balanced, width, label='Balanced', 
               alpha=0.7, color='lightblue')
        
        ax3.set_xlabel('Minority Class ID')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Minority Classes (<50 samples)')
        ax3.set_xticks(x_minority)
        ax3.set_xticklabels(minority_classes)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Metrics comparison
        ax4 = plt.subplot(3, 3, 4)
        metrics = ['Gini\nCoefficient', 'Min\nSamples', 'Max\nSamples\n(/100)', 'Minority\nClasses']
        
        original_metric_values = [
            original_metrics['gini_coefficient'],
            original_metrics['min_samples'],
            original_metrics['max_samples'] / 100,  # Scale down
            original_metrics['minority_classes']
        ]
        
        balanced_metric_values = [
            balanced_metrics['gini_coefficient'],
            balanced_metrics['min_samples'],
            balanced_metrics['max_samples'] / 100,  # Scale down
            balanced_metrics['minority_classes']
        ]
        
        x_metrics = np.arange(len(metrics))
        ax4.bar(x_metrics - width/2, original_metric_values, width, label='Original', 
               alpha=0.7, color='lightcoral')
        ax4.bar(x_metrics + width/2, balanced_metric_values, width, label='Balanced', 
               alpha=0.7, color='lightblue')
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('Dataset Quality Metrics')
        ax4.set_xticks(x_metrics)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Improvement analysis
        ax5 = plt.subplot(3, 3, 5)
        improvements = []
        improvement_labels = []
        
        for class_id in all_classes:
            original_count = original_counts.get(class_id, 0)
            balanced_count = balanced_counts.get(class_id, 0)
            if original_count > 0:
                improvement = (balanced_count - original_count) / original_count * 100
                improvements.append(improvement)
                improvement_labels.append(f'C{class_id}')
        
        ax5.bar(range(len(improvements)), improvements, alpha=0.7, color='lightgreen')
        ax5.set_xlabel('Class ID')
        ax5.set_ylabel('Improvement (%)')
        ax5.set_title('Sample Count Improvement by Class')
        ax5.set_xticks(range(len(improvements)))
        ax5.set_xticklabels(improvement_labels, rotation=45)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Distribution comparison
        ax6 = plt.subplot(3, 3, 6)
        original_dist = list(original_counts.values())
        balanced_dist = list(balanced_counts.values())
        
        ax6.hist(original_dist, bins=20, alpha=0.7, label='Original', color='lightcoral')
        ax6.hist(balanced_dist, bins=20, alpha=0.7, label='Balanced', color='lightblue')
        ax6.set_xlabel('Sample Count per Class')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Distribution of Sample Counts')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Summary statistics
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        
        summary_text = f"""
        AUGMENTATION SUMMARY:
        
        Original Dataset:
        • Total samples: {original_metrics['total_samples']:,}
        • Classes: {original_metrics['num_classes']}
        • Imbalance ratio: {original_metrics['imbalance_ratio']:.1f}:1
        • Gini coefficient: {original_metrics['gini_coefficient']:.3f}
        • Minority classes: {original_metrics['minority_classes']}
        
        Balanced Dataset:
        • Total samples: {balanced_metrics['total_samples']:,}
        • Classes: {balanced_metrics['num_classes']}
        • Imbalance ratio: {balanced_metrics['imbalance_ratio']:.1f}:1
        • Gini coefficient: {balanced_metrics['gini_coefficient']:.3f}
        • Minority classes: {balanced_metrics['minority_classes']}
        
        IMPROVEMENTS:
        • Sample increase: {balanced_metrics['total_samples'] - original_metrics['total_samples']:,}
        • Imbalance reduction: {original_metrics['imbalance_ratio'] / balanced_metrics['imbalance_ratio']:.1f}x
        • Gini improvement: {original_metrics['gini_coefficient'] - balanced_metrics['gini_coefficient']:.3f}
        """
        
        ax7.text(0.05, 0.95, summary_text, fontsize=10, transform=ax7.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Plot 8: Before/after sample counts
        ax8 = plt.subplot(3, 3, 8)
        ax8.scatter(original_values, balanced_values, alpha=0.6, s=50)
        ax8.plot([0, max(original_values)], [0, max(original_values)], 'r--', alpha=0.7, label='No change')
        ax8.set_xlabel('Original Sample Count')
        ax8.set_ylabel('Balanced Sample Count')
        ax8.set_title('Before vs After Sample Counts')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Quality metrics
        ax9 = plt.subplot(3, 3, 9)
        quality_metrics = ['Imbalance\nRatio', 'Gini\nCoefficient', 'Minority\nClasses']
        original_quality = [original_metrics['imbalance_ratio'], 
                          original_metrics['gini_coefficient'],
                          original_metrics['minority_classes']]
        balanced_quality = [balanced_metrics['imbalance_ratio'],
                          balanced_metrics['gini_coefficient'],
                          balanced_metrics['minority_classes']]
        
        x_quality = np.arange(len(quality_metrics))
        ax9.bar(x_quality - width/2, original_quality, width, label='Original', 
               alpha=0.7, color='lightcoral')
        ax9.bar(x_quality + width/2, balanced_quality, width, label='Balanced', 
               alpha=0.7, color='lightblue')
        
        ax9.set_xlabel('Quality Metrics')
        ax9.set_ylabel('Value')
        ax9.set_title('Dataset Quality Comparison')
        ax9.set_xticks(x_quality)
        ax9.set_xticklabels(quality_metrics)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the comprehensive comparison
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved comprehensive comparison: {save_path}")
        
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("📈 AUGMENTATION RESULTS SUMMARY")
        print("="*60)
        print(f"Original samples: {original_metrics['total_samples']:,}")
        print(f"Balanced samples: {balanced_metrics['total_samples']:,}")
        print(f"Sample increase: {balanced_metrics['total_samples'] - original_metrics['total_samples']:,}")
        print(f"Imbalance ratio improvement: {original_metrics['imbalance_ratio']:.1f}:1 → {balanced_metrics['imbalance_ratio']:.1f}:1")
        print(f"Gini coefficient improvement: {original_metrics['gini_coefficient']:.3f} → {balanced_metrics['gini_coefficient']:.3f}")
        print(f"Minority classes: {original_metrics['minority_classes']} → {balanced_metrics['minority_classes']}")
        
        # Save metrics to JSON for programmatic access
        metrics_data = {
            'original': original_metrics,
            'balanced': balanced_metrics,
            'improvement': {
                'sample_increase': balanced_metrics['total_samples'] - original_metrics['total_samples'],
                'imbalance_reduction': original_metrics['imbalance_ratio'] / balanced_metrics['imbalance_ratio'],
                'gini_improvement': original_metrics['gini_coefficient'] - balanced_metrics['gini_coefficient'],
                'minority_reduction': original_metrics['minority_classes'] - balanced_metrics['minority_classes']
            }
        }
        
        metrics_path = self.output_dir / f"{save_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"📊 Saved metrics: {metrics_path}")
    
    def generate_report(self, original_file: str, balanced_file: str) -> None:
        """Generate a comprehensive analysis report."""
        print("\n" + "="*60)
        print("📋 GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        # Load datasets
        original_df, original_counts, original_metrics = self.load_dataset(original_file)
        balanced_df, balanced_counts, balanced_metrics = self.load_dataset(balanced_file)
        
        if original_df is None or balanced_df is None:
            print("❌ Cannot generate report - datasets failed to load")
            return
        
        # Create individual plots
        self.plot_class_distribution(original_counts, "Original Dataset", "original_distribution")
        self.plot_class_distribution(balanced_counts, "Balanced Dataset", "balanced_distribution")
        
        # Create comprehensive comparison
        self.plot_before_after_comparison(original_file, balanced_file)
        
        print("\n✅ Analysis complete! Check the outputs/visualizations/ directory for all plots.")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CMCC-34 Data Augmentation Visualization")
    parser.add_argument("--original", default="../data/cmcc-34/train_new.csv", 
                       help="Path to original dataset")
    parser.add_argument("--balanced", default="data/train_balanced.csv", 
                       help="Path to balanced dataset")
    parser.add_argument("--output-dir", default="outputs/visualizations", 
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create visualizer and generate report
    visualizer = DataVisualizer(args.output_dir)
    visualizer.generate_report(args.original, args.balanced)


if __name__ == "__main__":
    main()