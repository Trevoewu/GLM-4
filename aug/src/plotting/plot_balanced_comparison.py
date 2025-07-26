#!/usr/bin/env python3
"""
Plot comparison between original and balanced CMCC-34 datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import os

def load_and_analyze_data(csv_file, dataset_name):
    """Load data and return analysis"""
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None, None
    
    df = pd.read_csv(csv_file)
    class_counts = Counter(df['c_numerical'])
    
    print(f"\n📊 {dataset_name} Dataset:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Number of classes: {len(class_counts)}")
    print(f"  Min samples per class: {min(class_counts.values())}")
    print(f"  Max samples per class: {max(class_counts.values())}")
    print(f"  Imbalance ratio: {max(class_counts.values())/min(class_counts.values()):.1f}:1")
    
    # Calculate Gini coefficient
    counts = list(class_counts.values())
    sorted_counts = np.sort(counts)
    n = len(counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum) / sum(counts)) / n
    print(f"  Gini coefficient: {gini:.3f}")
    
    return df, class_counts

def plot_comparison():
    """Plot comparison between original and balanced datasets"""
    
    # Load datasets
    original_file = "../data/cmcc-34/train_new.csv"
    balanced_file = "output/train_balanced.csv"
    
    print("Loading datasets for comparison...")
    original_df, original_counts = load_and_analyze_data(original_file, "Original")
    balanced_df, balanced_counts = load_and_analyze_data(balanced_file, "Balanced")
    
    if original_counts is None or balanced_counts is None:
        return
    
    # Prepare data for plotting
    all_classes = sorted(set(original_counts.keys()) | set(balanced_counts.keys()))
    orig_values = [original_counts.get(c, 0) for c in all_classes]
    bal_values = [balanced_counts.get(c, 0) for c in all_classes]
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Side-by-side comparison (log scale)
    x = np.arange(len(all_classes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, orig_values, width, label='Original', alpha=0.7, color='lightcoral')
    bars2 = ax1.bar(x + width/2, bal_values, width, label='Balanced', alpha=0.7, color='lightblue')
    
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Sample Count (log scale)')
    ax1.set_title('Original vs Balanced Dataset Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_classes, rotation=45)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement ratio
    improvements = []
    class_labels = []
    colors = []
    
    for i, c in enumerate(all_classes):
        orig = original_counts.get(c, 0)
        bal = balanced_counts.get(c, 0)
        if orig > 0:
            improvement = bal / orig
            improvements.append(improvement)
            class_labels.append(f'Class {c}')
            
            # Color coding based on improvement
            if improvement >= 10:
                colors.append('green')  # Significant improvement
            elif improvement >= 2:
                colors.append('orange')  # Moderate improvement
            else:
                colors.append('red')  # Little or no improvement
    
    bars = ax2.barh(class_labels, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Improvement Ratio (Balanced/Original)')
    ax2.set_title('Sample Count Improvement by Class')
    ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='No Change')
    ax2.axvline(x=2, color='blue', linestyle='--', alpha=0.5, label='2x Improvement')
    ax2.axvline(x=5, color='green', linestyle='--', alpha=0.5, label='5x Improvement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution shape comparison
    ax3.hist(orig_values, bins=20, alpha=0.5, label='Original', color='red', density=True)
    ax3.hist(bal_values, bins=20, alpha=0.5, label='Balanced', color='blue', density=True)
    ax3.set_xlabel('Sample Count')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution Shape Comparison')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics comparison
    stats_text = f"""Dataset Comparison Statistics:

ORIGINAL DATASET:
• Total samples: {sum(orig_values):,}
• Classes: {len(all_classes)}
• Min/Max: {min(orig_values)}/{max(orig_values):,}
• Mean: {np.mean(orig_values):.1f}
• Std: {np.std(orig_values):.1f}
• Imbalance: {max(orig_values)/min(orig_values):.1f}:1
• Gini: {calculate_gini(orig_values):.3f}

BALANCED DATASET:
• Total samples: {sum(bal_values):,}
• Classes: {len(all_classes)}
• Min/Max: {min(bal_values)}/{max(bal_values):,}
• Mean: {np.mean(bal_values):.1f}
• Std: {np.std(bal_values):.1f}
• Imbalance: {max(bal_values)/min(bal_values):.1f}:1
• Gini: {calculate_gini(bal_values):.3f}

IMPROVEMENT:
• Added samples: {sum(bal_values) - sum(orig_values):,}
• Gini reduction: {calculate_gini(orig_values) - calculate_gini(bal_values):.3f}
• Imbalance reduction: {(max(orig_values)/min(orig_values)) - (max(bal_values)/min(bal_values)):.1f}
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Comparison Statistics')
    
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs('output/plots', exist_ok=True)
    output_path = 'output/plots/balanced_vs_original_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {output_path}")
    plt.show()
    
    # Print detailed analysis
    print(f"\n🎯 Detailed Improvement Analysis:")
    print("=" * 60)
    
    # Find classes that were improved
    improved_classes = []
    for c in all_classes:
        orig = original_counts.get(c, 0)
        bal = balanced_counts.get(c, 0)
        if bal > orig:
            improved_classes.append((c, orig, bal, bal/orig if orig > 0 else float('inf')))
    
    improved_classes.sort(key=lambda x: x[3], reverse=True)  # Sort by improvement ratio
    
    print(f"Classes with sample augmentation ({len(improved_classes)} total):")
    for class_id, orig, bal, ratio in improved_classes:
        print(f"  Class {class_id:2d}: {orig:3d} → {bal:3d} samples ({ratio:5.1f}x improvement)")
    
    # Check if any minority classes still need attention
    minority_threshold = 50
    still_minority = [c for c in all_classes if balanced_counts.get(c, 0) < minority_threshold]
    
    if still_minority:
        print(f"\n⚠️  Classes still needing attention (<{minority_threshold} samples):")
        for c in still_minority:
            print(f"  Class {c:2d}: {balanced_counts.get(c, 0)} samples")
    else:
        print(f"\n🎉 All classes now have ≥{minority_threshold} samples!")

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
    plot_comparison() 