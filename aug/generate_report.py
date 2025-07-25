#!/usr/bin/env python3
"""
Generate detailed data augmentation report for CMCC-34 dataset
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os

def generate_comprehensive_report():
    """Generate a comprehensive data augmentation report"""
    
    # Load datasets
    original_df = pd.read_csv("../data/cmcc-34/train_new.csv")
    balanced_df = pd.read_csv("output/train_balanced.csv")
    
    original_counts = Counter(original_df['c_numerical'])
    balanced_counts = Counter(balanced_df['c_numerical'])
    
    # Business type mapping
    business_types = {
        0: "咨询（含查询）业务规定", 1: "办理取消", 2: "咨询（含查询）业务资费",
        3: "咨询（含查询）营销活动信息", 4: "咨询（含查询）办理方式", 5: "投诉（含抱怨）业务使用问题",
        6: "咨询（含查询）账户信息", 7: "办理开通", 8: "咨询（含查询）业务订购信息查询",
        9: "投诉（含抱怨）不知情定制问题", 10: "咨询（含查询）产品/业务功能", 11: "咨询（含查询）用户资料",
        12: "投诉（含抱怨）费用问题", 13: "投诉（含抱怨）业务办理问题", 14: "投诉（含抱怨）服务问题",
        15: "办理变更", 16: "咨询（含查询）服务渠道信息", 17: "投诉（含抱怨）业务规定不满",
        18: "投诉（含抱怨）营销问题", 19: "投诉（含抱怨）网络问题", 20: "办理停复机",
        21: "投诉（含抱怨）信息安全问题", 22: "办理重置/修改/补发", 23: "咨询（含查询）使用方式",
        24: "咨询（含查询）号码状态", 25: "咨询（含查询）工单处理结果", 26: "办理打印/邮寄",
        27: "咨询（含查询）宽带覆盖范围", 28: "办理移机/装机/拆机", 29: "办理缴费",
        30: "办理下载/设置", 31: "办理补换卡", 32: "办理销户/重开", 33: "咨询（含查询）电商货品信息"
    }
    
    def calculate_gini(counts):
        """Calculate Gini coefficient"""
        counts = np.array(list(counts.values()))
        n = len(counts)
        total = np.sum(counts)
        sorted_counts = np.sort(counts)
        cumsum = np.cumsum(sorted_counts)
        return (n + 1 - 2 * np.sum(cumsum) / total) / n
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("                CMCC-34 DATA AUGMENTATION REPORT")
    report.append("=" * 80)
    
    # Overview
    report.append("\n📊 DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"Original dataset: {len(original_df):,} samples, {len(original_counts)} classes")
    report.append(f"Balanced dataset: {len(balanced_df):,} samples, {len(balanced_counts)} classes")
    report.append(f"Added samples: {len(balanced_df) - len(original_df):,}")
    report.append(f"Augmentation rate: {((len(balanced_df) - len(original_df)) / len(original_df) * 100):.2f}%")
    
    # Class imbalance metrics
    orig_gini = calculate_gini(original_counts)
    bal_gini = calculate_gini(balanced_counts)
    orig_ratio = max(original_counts.values()) / min(original_counts.values())
    bal_ratio = max(balanced_counts.values()) / min(balanced_counts.values())
    
    report.append(f"\n📈 IMBALANCE IMPROVEMENT")
    report.append("-" * 40)
    report.append(f"Gini coefficient: {orig_gini:.3f} → {bal_gini:.3f} (Δ{orig_gini - bal_gini:+.3f})")
    report.append(f"Imbalance ratio: {orig_ratio:.1f}:1 → {bal_ratio:.1f}:1")
    report.append(f"Standard deviation: {np.std(list(original_counts.values())):.1f} → {np.std(list(balanced_counts.values())):.1f}")
    
    # Detailed class analysis
    report.append(f"\n🎯 DETAILED CLASS ANALYSIS")
    report.append("-" * 40)
    
    all_classes = sorted(set(original_counts.keys()) | set(balanced_counts.keys()))
    
    # Categories by improvement level
    significant_improvement = []  # >5x
    moderate_improvement = []     # 2-5x
    slight_improvement = []       # 1.1-2x
    no_change = []               # 1x
    
    for class_id in all_classes:
        orig = original_counts.get(class_id, 0)
        bal = balanced_counts.get(class_id, 0)
        
        if orig > 0:
            ratio = bal / orig
            class_info = (class_id, orig, bal, ratio, business_types.get(class_id, f"Unknown {class_id}"))
            
            if ratio >= 5:
                significant_improvement.append(class_info)
            elif ratio >= 2:
                moderate_improvement.append(class_info)
            elif ratio > 1.1:
                slight_improvement.append(class_info)
            else:
                no_change.append(class_info)
    
    # Report improvements
    if significant_improvement:
        report.append(f"\n🚀 SIGNIFICANT IMPROVEMENT (≥5x):")
        for class_id, orig, bal, ratio, name in sorted(significant_improvement, key=lambda x: x[3], reverse=True):
            report.append(f"  Class {class_id:2d}: {orig:3d} → {bal:3d} ({ratio:4.1f}x) - {name}")
    
    if moderate_improvement:
        report.append(f"\n📈 MODERATE IMPROVEMENT (2-5x):")
        for class_id, orig, bal, ratio, name in sorted(moderate_improvement, key=lambda x: x[3], reverse=True):
            report.append(f"  Class {class_id:2d}: {orig:3d} → {bal:3d} ({ratio:4.1f}x) - {name}")
    
    if slight_improvement:
        report.append(f"\n📊 SLIGHT IMPROVEMENT (1.1-2x):")
        for class_id, orig, bal, ratio, name in sorted(slight_improvement, key=lambda x: x[3], reverse=True):
            report.append(f"  Class {class_id:2d}: {orig:3d} → {bal:3d} ({ratio:4.1f}x) - {name}")
    
    # Classes still needing attention
    minority_classes = [(c, balanced_counts.get(c, 0), business_types.get(c, f"Unknown {c}")) 
                       for c in all_classes if balanced_counts.get(c, 0) < 50]
    minority_classes.sort(key=lambda x: x[1])
    
    if minority_classes:
        report.append(f"\n⚠️  CLASSES STILL NEEDING ATTENTION (<50 samples):")
        for class_id, count, name in minority_classes:
            priority = "🔴 URGENT" if count < 10 else "🟡 HIGH" if count < 25 else "🟠 MEDIUM"
            report.append(f"  {priority} Class {class_id:2d}: {count:2d} samples - {name}")
    
    # Synthetic samples analysis
    if 'synthetic' in balanced_df.columns:
        synthetic_samples = balanced_df[balanced_df['synthetic'] == True]
        report.append(f"\n🤖 SYNTHETIC SAMPLES ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total synthetic samples: {len(synthetic_samples):,}")
        report.append(f"Percentage of dataset: {len(synthetic_samples) / len(balanced_df) * 100:.2f}%")
        
        synthetic_by_class = Counter(synthetic_samples['c_numerical'])
        if synthetic_by_class:
            report.append(f"Classes with synthetic samples:")
            for class_id, count in sorted(synthetic_by_class.items()):
                report.append(f"  Class {class_id:2d}: {count:2d} synthetic samples")
    
    # Recommendations
    report.append(f"\n💡 RECOMMENDATIONS FOR FURTHER IMPROVEMENT")
    report.append("-" * 40)
    
    if len(minority_classes) > 0:
        report.append(f"1. Focus on {len([c for c in minority_classes if c[1] < 10])} ultra-minority classes (<10 samples)")
        report.append(f"2. Target 100-200 samples per class for optimal balance")
        report.append(f"3. Expected Gini coefficient improvement: {bal_gini:.3f} → ~0.300")
    
    report.append(f"4. Consider advanced augmentation techniques:")
    report.append(f"   - Back-translation for paraphrasing")
    report.append(f"   - Template-based generation")
    report.append(f"   - Few-shot learning with LLMs")
    
    report.append(f"\n📋 NEXT STEPS")
    report.append("-" * 40)
    report.append(f"1. Train model with balanced dataset")
    report.append(f"2. Evaluate performance on minority classes")
    report.append(f"3. Compare F1-scores: macro, micro, weighted")
    report.append(f"4. Iterate augmentation strategy based on results")
    
    report.append("\n" + "=" * 80)
    report.append("Report generated by GLM-4 Data Augmentation Tool")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    report_path = "output/data_augmentation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Detailed report saved: {report_path}")
    
    # Generate JSON summary for programmatic use
    summary = {
        "original": {
            "total_samples": len(original_df),
            "classes": len(original_counts),
            "gini_coefficient": float(orig_gini),
            "imbalance_ratio": float(orig_ratio),
            "min_samples": int(min(original_counts.values())),
            "max_samples": int(max(original_counts.values()))
        },
        "balanced": {
            "total_samples": len(balanced_df),
            "classes": len(balanced_counts),
            "gini_coefficient": float(bal_gini),
            "imbalance_ratio": float(bal_ratio),
            "min_samples": int(min(balanced_counts.values())),
            "max_samples": int(max(balanced_counts.values()))
        },
        "improvement": {
            "added_samples": len(balanced_df) - len(original_df),
            "gini_reduction": float(orig_gini - bal_gini),
            "classes_improved": len(significant_improvement) + len(moderate_improvement) + len(slight_improvement),
            "classes_needing_attention": len(minority_classes)
        }
    }
    
    summary_path = "output/augmentation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Machine-readable summary saved: {summary_path}")

if __name__ == "__main__":
    generate_comprehensive_report() 