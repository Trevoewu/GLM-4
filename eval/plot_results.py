#!/usr/bin/env python3
"""
Script to plot the distribution of quality scores from evaluation results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List


def load_results(results_file: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_quality_distribution(results: List[Dict], output_file: str = None):
    """Plot the distribution of quality scores."""
    # Extract quality scores
    quality_scores = [r['quality_score'] for r in results if r['quality_score'] > 0]
    
    if not quality_scores:
        print("No valid quality scores found")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Synthetic Dialogue Quality Evaluation Results', fontsize=16)
    
    # 1. Quality Score Distribution
    ax1.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Quality Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Quality Score Distribution')
    ax1.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(quality_scores):.2f}')
    ax1.axvline(np.median(quality_scores), color='green', linestyle='--', 
                label=f'Median: {np.median(quality_scores):.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Score Categories
    categories = {
        'Excellent (9-10)': len([s for s in quality_scores if s >= 9]),
        'Good (7-8)': len([s for s in quality_scores if 7 <= s < 9]),
        'Fair (5-6)': len([s for s in quality_scores if 5 <= s < 7]),
        'Poor (3-4)': len([s for s in quality_scores if 3 <= s < 5]),
        'Very Poor (1-2)': len([s for s in quality_scores if 1 <= s < 3])
    }
    
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    ax2.bar(categories.keys(), categories.values(), color=colors, alpha=0.7)
    ax2.set_title('Quality Score Categories')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    total = len(quality_scores)
    for i, (category, count) in enumerate(categories.items()):
        percentage = (count / total) * 100
        ax2.text(i, count + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 3. Dimension Analysis
    semantic_scores = [r['dimension_analysis']['semantic_consistency']['score'] 
                      for r in results if r['dimension_analysis']['semantic_consistency']['score'] > 0]
    context_scores = [r['dimension_analysis']['context_completeness']['score'] 
                     for r in results if r['dimension_analysis']['context_completeness']['score'] > 0]
    language_scores = [r['dimension_analysis']['language_naturalness']['score'] 
                      for r in results if r['dimension_analysis']['language_naturalness']['score'] > 0]
    
    dimensions = ['Semantic\nConsistency', 'Context\nCompleteness', 'Language\nNaturalness']
    means = [np.mean(semantic_scores) if semantic_scores else 0,
             np.mean(context_scores) if context_scores else 0,
             np.mean(language_scores) if language_scores else 0]
    
    bars = ax3.bar(dimensions, means, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    ax3.set_title('Average Scores by Dimension')
    ax3.set_ylabel('Average Score')
    ax3.set_ylim(0, 10)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean:.2f}', ha='center', va='bottom')
    
    # 4. Confidence vs Quality Score
    confidence_scores = [r['confidence'] for r in results if r['confidence'] > 0]
    valid_indices = [i for i, r in enumerate(results) if r['quality_score'] > 0 and r['confidence'] > 0]
    valid_quality = [results[i]['quality_score'] for i in valid_indices]
    valid_confidence = [results[i]['confidence'] for i in valid_indices]
    
    if valid_quality and valid_confidence:
        scatter = ax4.scatter(valid_quality, valid_confidence, alpha=0.6, c='purple')
        ax4.set_xlabel('Quality Score')
        ax4.set_ylabel('Confidence')
        ax4.set_title('Quality Score vs Confidence')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(valid_quality, valid_confidence)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()


def print_statistics(results: List[Dict]):
    """Print detailed statistics."""
    quality_scores = [r['quality_score'] for r in results if r['quality_score'] > 0]
    
    if not quality_scores:
        print("No valid quality scores found")
        return
    
    print("\n" + "="*60)
    print("DETAILED QUALITY SCORE STATISTICS")
    print("="*60)
    print(f"Total samples: {len(results)}")
    print(f"Valid scores: {len(quality_scores)}")
    print(f"Mean quality score: {np.mean(quality_scores):.2f}")
    print(f"Median quality score: {np.median(quality_scores):.2f}")
    print(f"Standard deviation: {np.std(quality_scores):.2f}")
    print(f"Min score: {np.min(quality_scores):.1f}")
    print(f"Max score: {np.max(quality_scores):.1f}")
    
    # Score distribution
    print("\nScore Distribution:")
    categories = {
        'Excellent (9-10)': len([s for s in quality_scores if s >= 9]),
        'Good (7-8)': len([s for s in quality_scores if 7 <= s < 9]),
        'Fair (5-6)': len([s for s in quality_scores if 5 <= s < 7]),
        'Poor (3-4)': len([s for s in quality_scores if 3 <= s < 5]),
        'Very Poor (1-2)': len([s for s in quality_scores if 1 <= s < 3])
    }
    
    for category, count in categories.items():
        percentage = (count / len(quality_scores)) * 100
        print(f"  {category}: {count} samples ({percentage:.1f}%)")
    
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Plot quality score distribution from evaluation results")
    parser.add_argument("--results_file", default="output_synthetic_quality/batch_results.json", 
                       help="Path to evaluation results JSON file")
    parser.add_argument("--output_plot", default="output_synthetic_quality/quality_distribution.png", 
                       help="Output plot file path")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    data = load_results(args.results_file)
    results = data.get('detailed_results', [])
    
    if not results:
        print("No results found in the file")
        return
    
    print(f"Loaded {len(results)} evaluation results")
    
    # Print statistics
    print_statistics(results)
    
    # Create plots
    print(f"Creating plots...")
    plot_quality_distribution(results, args.output_plot)


if __name__ == "__main__":
    main()
