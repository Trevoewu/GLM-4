#!/usr/bin/env python3
"""
Final comparison: Original vs Balanced vs Enhanced datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import os

def plot_three_way_comparison():
    """Plot comparison between original, balanced, and enhanced datasets"""
    
    # Load all three datasets
    datasets = {
        'Original': '../data/cmcc-34/train_new.csv',
        'Balanced (GLM-4)': 'output/train_balanced.csv', 
        'Enhanced (Templates)': 'output/train_enhanced.csv'
    }
    
    data_analysis = {}
    
    print("📊 Loading datasets for three-way comparison...")
    
    for name, path in datasets.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            counts = Counter(df['c_numerical'])
            
            # Calculate metrics
            gini = calculate_gini(list(counts.values()))
            imbalance_ratio = max(counts.values()) / min(counts.values())
            
            data_analysis[name] = {
                'df': df,
                'counts': counts,
                'total_samples': len(df),
                'gini': gini,
                'imbalance_ratio': imbalance_ratio,
                'min_samples': min(counts.values()),
                'max_samples': max(counts.values())
            }
            
            print(f"{name}: {len(df):,} samples, Gini: {gini:.3f}, Ratio: {imbalance_ratio:.1f}:1")
    
    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(20, 12))
    
    # Get all classes
    all_classes = sorted(set().union(*[d['counts'].keys() for d in data_analysis.values()]))
    
    # Plot 1: Side-by-side comparison (4x2 layout)
    ax1 = plt.subplot(2, 4, 1)
    width = 0.25
    x = np.arange(len(all_classes))
    
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for i, (name, data) in enumerate(data_analysis.items()):
        values = [data['counts'].get(c, 0) for c in all_classes]
        ax1.bar(x + i*width, values, width, label=name, alpha=0.7, color=colors[i])
    
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Sample Count (log scale)')
    ax1.set_title('Sample Count Comparison (Log Scale)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(all_classes, rotation=45)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Focus on minority classes
    ax2 = plt.subplot(2, 4, 2)
    minority_classes = [c for c in all_classes if data_analysis['Original']['counts'].get(c, 0) < 50]
    
    for i, (name, data) in enumerate(data_analysis.items()):
        values = [data['counts'].get(c, 0) for c in minority_classes]
        ax2.bar(np.arange(len(minority_classes)) + i*width, values, width, 
                label=name, alpha=0.7, color=colors[i])
    
    ax2.set_xlabel('Minority Class ID')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Minority Classes (<50 samples)')
    ax2.set_xticks(np.arange(len(minority_classes)) + width)
    ax2.set_xticklabels(minority_classes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Metrics comparison
    ax3 = plt.subplot(2, 4, 3)
    metrics = ['Gini Coefficient', 'Min Samples', 'Max Samples']
    
    gini_values = [data_analysis[name]['gini'] for name in data_analysis.keys()]
    min_values = [data_analysis[name]['min_samples'] for name in data_analysis.keys()]
    max_values = [data_analysis[name]['max_samples']/100 for name in data_analysis.keys()]  # Scale down
    
    x_pos = np.arange(len(data_analysis))
    
    ax3_twin = ax3.twinx()
    
    bars1 = ax3.bar(x_pos - 0.25, gini_values, 0.25, label='Gini Coefficient', color='red', alpha=0.7)
    bars2 = ax3.bar(x_pos, min_values, 0.25, label='Min Samples', color='blue', alpha=0.7)
    bars3 = ax3_twin.bar(x_pos + 0.25, max_values, 0.25, label='Max Samples (÷100)', color='green', alpha=0.7)
    
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Gini / Min Samples')
    ax3_twin.set_ylabel('Max Samples (÷100)')
    ax3.set_title('Key Metrics Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(data_analysis.keys(), rotation=45)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 4: Distribution shapes
    ax4 = plt.subplot(2, 4, 4)
    
    for name, data in data_analysis.items():
        values = list(data['counts'].values())
        ax4.hist(values, bins=20, alpha=0.5, label=name, density=True)
    
    ax4.set_xlabel('Sample Count')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution Shape Comparison')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5-8: Detailed statistics
    ax5 = plt.subplot(2, 4, (5, 8))
    
    # Create detailed statistics text
    stats_text = "COMPREHENSIVE DATASET COMPARISON\n"
    stats_text += "=" * 80 + "\n\n"
    
    for name, data in data_analysis.items():
        stats_text += f"{name.upper()}:\n"
        stats_text += f"  • Total samples: {data['total_samples']:,}\n"
        stats_text += f"  • Classes: {len(data['counts'])}\n"
        stats_text += f"  • Min/Max samples: {data['min_samples']}/{data['max_samples']:,}\n"
        stats_text += f"  • Mean samples per class: {np.mean(list(data['counts'].values())):.1f}\n"
        stats_text += f"  • Std deviation: {np.std(list(data['counts'].values())):.1f}\n"
        stats_text += f"  • Gini coefficient: {data['gini']:.3f}\n"
        stats_text += f"  • Imbalance ratio: {data['imbalance_ratio']:.1f}:1\n\n"
    
    # Calculate improvements
    if 'Enhanced (Templates)' in data_analysis and 'Original' in data_analysis:
        orig = data_analysis['Original']
        enhanced = data_analysis['Enhanced (Templates)']
        
        stats_text += "IMPROVEMENT ANALYSIS:\n"
        stats_text += f"  • Added samples: {enhanced['total_samples'] - orig['total_samples']:,}\n"
        stats_text += f"  • Gini reduction: {orig['gini'] - enhanced['gini']:+.3f}\n"
        stats_text += f"  • Imbalance improvement: {orig['imbalance_ratio']:.1f}:1 → {enhanced['imbalance_ratio']:.1f}:1\n"
        stats_text += f"  • Min samples improvement: {orig['min_samples']} → {enhanced['min_samples']}\n\n"
    
    # Ultra-minority analysis
    ultra_minority = [33, 32, 29, 30, 31]
    stats_text += "ULTRA-MINORITY CLASSES PROGRESS:\n"
    for class_id in ultra_minority:
        orig_count = data_analysis['Original']['counts'].get(class_id, 0)
        if 'Enhanced (Templates)' in data_analysis:
            enh_count = data_analysis['Enhanced (Templates)']['counts'].get(class_id, 0)
            improvement = enh_count / orig_count if orig_count > 0 else float('inf')
            stats_text += f"  • Class {class_id:2d}: {orig_count:3d} → {enh_count:3d} ({improvement:5.1f}x)\n"
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs('output/plots', exist_ok=True)
    output_path = 'output/plots/final_three_way_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Three-way comparison plot saved: {output_path}")
    
    # Print summary
    print(f"\n🎯 FINAL AUGMENTATION SUMMARY:")
    print("=" * 60)
    
    orig = data_analysis['Original']
    enhanced = data_analysis['Enhanced (Templates)']
    
    print(f"📊 Dataset Growth:")
    print(f"   Original → Enhanced: {orig['total_samples']:,} → {enhanced['total_samples']:,} samples")
    print(f"   Growth rate: {(enhanced['total_samples']/orig['total_samples']-1)*100:.1f}%")
    
    print(f"\n📈 Balance Improvement:")
    print(f"   Gini coefficient: {orig['gini']:.3f} → {enhanced['gini']:.3f}")
    print(f"   Imbalance ratio: {orig['imbalance_ratio']:.1f}:1 → {enhanced['imbalance_ratio']:.1f}:1")
    print(f"   Min samples per class: {orig['min_samples']} → {enhanced['min_samples']}")
    
    print(f"\n🎉 Success Metrics:")
    minority_resolved = sum(1 for c in ultra_minority 
                          if enhanced['counts'].get(c, 0) >= 50)
    print(f"   Ultra-minority classes resolved: {minority_resolved}/5")
    print(f"   Remaining classes <50 samples: {sum(1 for c in enhanced['counts'].values() if c < 50)}")
    
    if enhanced['gini'] < 0.4:
        print(f"   🏆 EXCELLENT: Gini coefficient < 0.4 achieved!")
    elif enhanced['gini'] < 0.5:
        print(f"   ✅ GOOD: Significant imbalance reduction achieved!")
    else:
        print(f"   📈 PROGRESS: Further augmentation recommended")

def calculate_gini(counts):
    """Calculate Gini coefficient"""
    counts = np.array(counts)
    n = len(counts)
    total = np.sum(counts)
    
    if total == 0:
        return 0
    
    sorted_counts = np.sort(counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum) / total) / n
    
    return gini

if __name__ == "__main__":
    plot_three_way_comparison() 