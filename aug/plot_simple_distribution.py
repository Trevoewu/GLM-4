#!/usr/bin/env python3
"""
Simple English version of CMCC-34 class distribution plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import os

def plot_simple_distribution():
    """Plot simple class distribution in English"""
    # Load data
    csv_file = "../data/cmcc-34/train_new.csv"
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    class_counts = Counter(df['c_numerical'])
    
    # Prepare data
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Bar chart
    bars = ax1.bar(classes, counts, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title('CMCC-34 Original Class Distribution')
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Sample Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Log scale bar chart
    log_counts = [np.log10(c + 1) for c in counts]
    ax2.bar(classes, log_counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_title('CMCC-34 Distribution (Log Scale)')
    ax2.set_xlabel('Class ID')
    ax2.set_ylabel('Sample Count (log10)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Top 10 classes pie chart
    top_10_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_counts = [item[1] for item in top_10_classes]
    top_10_labels = [f'Class {item[0]}' for item in top_10_classes]
    
    ax3.pie(top_10_counts, labels=top_10_labels, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Top 10 Classes Distribution')
    
    # Plot 4: Statistics
    total_samples = sum(counts)
    max_samples = max(counts)
    min_samples = min(counts)
    mean_samples = np.mean(counts)
    std_samples = np.std(counts)
    imbalance_ratio = max_samples / min_samples
    
    # Calculate Gini coefficient
    sorted_counts = np.sort(counts)
    n = len(counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum) / total_samples) / n
    
    stats_text = f"""Dataset Statistics:
    
Total Samples: {total_samples:,}
Number of Classes: {len(classes)}
Max Samples: {max_samples:,}
Min Samples: {min_samples:,}
Mean Samples: {mean_samples:.1f}
Std Deviation: {std_samples:.1f}
Imbalance Ratio: {imbalance_ratio:.1f}:1
Gini Coefficient: {gini:.3f}

Class Imbalance Severity:
- Extremely imbalanced (>100:1)
- Needs urgent balancing
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Dataset Statistics')
    
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/cmcc34_class_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Class distribution plot saved: output/plots/cmcc34_class_distribution.png")
    plt.show()
    
    # Print detailed statistics
    print("\n📊 CMCC-34 Dataset Analysis:")
    print("=" * 50)
    print(f"Total samples: {total_samples:,}")
    print(f"Number of classes: {len(classes)}")
    print(f"Most frequent class: {max(class_counts, key=class_counts.get)} ({max_samples:,} samples)")
    print(f"Least frequent classes: {[k for k, v in class_counts.items() if v == min_samples]} ({min_samples} sample{'s' if min_samples > 1 else ''})")
    print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"Gini coefficient: {gini:.3f}")
    
    # Show minority classes (< 50 samples)
    minority_classes = [(k, v) for k, v in class_counts.items() if v < 50]
    minority_classes.sort(key=lambda x: x[1])
    
    print(f"\n🎯 Minority classes ({len(minority_classes)} classes with <50 samples):")
    for class_id, count in minority_classes:
        print(f"  Class {class_id:2d}: {count:3d} samples")
    
    print(f"\n💡 Recommended augmentation strategy:")
    print(f"  - Target samples per class: 100-200")
    print(f"  - Focus on classes with <10 samples first")
    print(f"  - Expected improvement: Gini coefficient from {gini:.3f} to ~0.3")

if __name__ == "__main__":
    plot_simple_distribution() 