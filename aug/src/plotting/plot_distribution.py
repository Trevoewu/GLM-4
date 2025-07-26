#!/usr/bin/env python3
"""
绘制CMCC-34数据集类别分布图
支持对比原始数据和平衡后数据的分布
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(csv_file):
    """加载数据并统计类别分布"""
    if not os.path.exists(csv_file):
        print(f"文件不存在: {csv_file}")
        return None, None
    
    df = pd.read_csv(csv_file)
    print(f"加载数据: {csv_file}")
    print(f"总样本数: {len(df)}")
    
    # 统计类别分布
    class_counts = Counter(df['c_numerical'])
    return df, class_counts

def plot_class_distribution(class_counts, title, save_path=None):
    """绘制类别分布图"""
    # 准备数据
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    # 创建图形
    plt.figure(figsize=(16, 10))
    
    # 子图1: 柱状图
    plt.subplot(2, 2, 1)
    bars = plt.bar(classes, counts, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.title(f'{title} - 柱状图')
    plt.xlabel('类别ID')
    plt.ylabel('样本数')
    plt.xticks(rotation=45)
    
    # 在柱子上显示数值
    for bar, count in zip(bars, counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontsize=8)
    
    # 子图2: 对数柱状图
    plt.subplot(2, 2, 2)
    log_counts = [np.log10(c + 1) for c in counts]  # +1避免log(0)
    plt.bar(classes, log_counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
    plt.title(f'{title} - 对数柱状图')
    plt.xlabel('类别ID')
    plt.ylabel('样本数 (log10)')
    plt.xticks(rotation=45)
    
    # 子图3: 饼图（显示前10大类别）
    plt.subplot(2, 2, 3)
    top_10_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_counts = [item[1] for item in top_10_classes]
    top_10_labels = [f'类别{item[0]}' for item in top_10_classes]
    
    plt.pie(top_10_counts, labels=top_10_labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'{title} - 前10大类别分布')
    
    # 子图4: 分布统计
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.9, f"总样本数: {sum(counts):,}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f"类别数: {len(classes)}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"最大样本数: {max(counts):,}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"最小样本数: {min(counts):,}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"平均样本数: {np.mean(counts):.1f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"标准差: {np.std(counts):.1f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f"不平衡比例: {max(counts)/min(counts):.1f}:1", fontsize=12, transform=plt.gca().transAxes)
    
    # 计算基尼系数
    gini = calculate_gini(counts)
    plt.text(0.1, 0.2, f"基尼系数: {gini:.3f}", fontsize=12, transform=plt.gca().transAxes)
    
    plt.axis('off')
    plt.title('统计信息')
    
    plt.tight_layout()
    
    if save_path:
        # Ensure output directory exists
        os.makedirs('output/plots', exist_ok=True)
        output_path = f'output/plots/{save_path}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()

def calculate_gini(counts):
    """计算基尼系数衡量不平衡程度"""
    counts = np.array(counts)
    n = len(counts)
    total = np.sum(counts)
    
    if total == 0:
        return 0
    
    # 排序
    sorted_counts = np.sort(counts)
    
    # 计算基尼系数
    cumsum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum) / total) / n
    
    return gini

def compare_distributions(original_counts, balanced_counts):
    """对比原始和平衡后的分布"""
    classes = sorted(set(original_counts.keys()) | set(balanced_counts.keys()))
    
    orig_counts = [original_counts.get(c, 0) for c in classes]
    bal_counts = [balanced_counts.get(c, 0) for c in classes]
    
    plt.figure(figsize=(15, 8))
    
    # 对比柱状图
    x = np.arange(len(classes))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, orig_counts, width, label='原始数据', alpha=0.7, color='lightcoral')
    plt.bar(x + width/2, bal_counts, width, label='平衡后数据', alpha=0.7, color='lightblue')
    
    plt.xlabel('类别ID')
    plt.ylabel('样本数')
    plt.title('类别分布对比')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.yscale('log')  # 使用对数刻度
    
    # 改善效果图
    plt.subplot(1, 2, 2)
    improvements = []
    class_labels = []
    
    for c in classes:
        orig = original_counts.get(c, 0)
        bal = balanced_counts.get(c, 0)
        if orig > 0:
            improvement = bal / orig
            improvements.append(improvement)
            class_labels.append(f'类别{c}')
    
    colors = ['red' if imp < 2 else 'orange' if imp < 5 else 'green' for imp in improvements]
    
    plt.barh(class_labels, improvements, color=colors, alpha=0.7)
    plt.xlabel('增强倍数')
    plt.title('各类别样本增强效果')
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='无增强线')
    plt.legend()
    
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("CMCC-34数据集类别分布分析")
    print("=" * 50)
    
    # 原始训练数据
    original_file = "../data/cmcc-34/train_new.csv"
    original_df, original_counts = load_data(original_file)
    
    if original_counts:
        print("\n📊 绘制原始数据分布图...")
        plot_class_distribution(original_counts, "CMCC-34原始训练数据", "original_distribution.png")
    
    # 平衡后数据（如果存在）
    balanced_file = "train_balanced.csv"
    if os.path.exists(balanced_file):
        print("\n📊 绘制平衡后数据分布图...")
        balanced_df, balanced_counts = load_data(balanced_file)
        if balanced_counts:
            plot_class_distribution(balanced_counts, "CMCC-34平衡后训练数据", "balanced_distribution.png")
            
            # 对比分析
            print("\n📊 绘制对比分析图...")
            compare_distributions(original_counts, balanced_counts)
    else:
        print(f"\n⚠️  平衡后数据文件不存在: {balanced_file}")
        print("请先运行数据增强: python run_augmentation.py")
    
    print("\n✅ 分布图绘制完成！")

if __name__ == "__main__":
    main() 