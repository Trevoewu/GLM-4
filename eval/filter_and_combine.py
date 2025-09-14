#!/usr/bin/env python3
"""
Simple script to filter synthetic data and combine with original data.
Follows KISS principle - Keep It Simple, Stupid.
"""

import json
import os
import sys
from pathlib import Path


def load_jsonl(file_path):
    """Load JSONL file."""
    print(f"Loading JSONL from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from JSONL")
    return data


def load_json(file_path):
    """Load JSON file."""
    print(f"Loading JSON from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded JSON with {len(data)} items")
    return data


def load_data_file(file_path):
    """Load data file that could be JSON or JSONL format."""
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Check if it's a JSON array (starts with [)
    if content.startswith('['):
        data = json.loads(content)
        print(f"Loaded JSON array with {len(data)} items")
        return data
    else:
        # Try JSONL format
        data = []
        for line in content.split('\n'):
            if line.strip():
                data.append(json.loads(line))
        print(f"Loaded JSONL with {len(data)} items")
        return data


def filter_synthetic_data(eval_results, quality_threshold=7.0, confidence_threshold=0.8):
    """Filter synthetic data based on quality_score and confidence."""
    print(f"Filtering with quality_threshold={quality_threshold}, confidence_threshold={confidence_threshold}")
    filtered = []
    for result in eval_results:
        quality_score = result.get('quality_score', 0)
        confidence = result.get('confidence', 0.0)
        
        if quality_score >= quality_threshold and confidence >= confidence_threshold:
            # Extract original synthetic data
            synthetic_sample = {
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是客服意图识别专家。分析对话内容，判断用户最终意图。\n\n对话格式：多个说话轮次用[SEP]分隔，通常以"您好请讲"开始。\n\n判断标准：\n- 关注用户最终目标，不是中间过程\n- 结合关键词和上下文综合判断\n\n业务类型列表：\n0:咨询（含查询）业务规定 1:办理取消 2:咨询（含查询）业务资费 3:咨询（含查询）营销活动信息 4:咨询（含查询）办理方式\n5:投诉（含抱怨）业务使用问题 6:咨询（含查询）账户信息 7:办理开通 8:咨询（含查询）业务订购信息查询 9:投诉（含抱怨）不知情定制问题\n10:咨询（含查询）产品/业务功能 11:咨询（含查询）用户资料 12:投诉（含抱怨）费用问题 13:投诉（含抱怨）业务办理问题 14:投诉（含抱怨）服务问题\n15:办理变更 16:咨询（含查询）服务渠道信息 17:投诉（含抱怨）业务规定不满 18:投诉（含抱怨）营销问题 19:投诉（含抱怨）网络问题\n20:办理停复机 21:投诉（含抱怨）信息安全问题 22:办理重置/修改/补发 23:咨询（含查询）使用方式 24:咨询（含查询）号码状态\n25:咨询（含查询）工单处理结果 26:办理打印/邮寄 27:咨询（含查询）宽带覆盖范围 28:办理移机/装机/拆机 29:办理缴费\n30:办理下载/设置 31:办理补换卡 32:办理销户/重开 33:咨询（含查询）电商货品信息\n\n输出格式：\n意图：[选择最合适的意图]'
                    },
                    {
                        'role': 'user',
                        'content': f'对话：{result.get("original_text", "")}'
                    },
                    {
                        'role': 'assistant',
                        'content': f'意图：{result.get("label", "")}'
                    }
                ]
            }
            filtered.append(synthetic_sample)
    
    print(f"Filtered {len(filtered)} synthetic samples")
    return filtered


def main():
    """Main function."""
    try:
        # Parameters
        quality_threshold = 8.0
        confidence_threshold = 0.8
        
        # Input files
        eval_file = "output_synthetic_quality/batch_results.json"
        train_file = "../data/cmcc-34/train.jsonl"
        dev_file = "../data/cmcc-34/dev.jsonl"
        
        # Output directory
        output_dir = "../data/filtered/"
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading evaluation results...")
        eval_data = load_json(eval_file)
        eval_results = eval_data.get('detailed_results', eval_data)
        
        print("Filtering synthetic data...")
        filtered_synthetic = filter_synthetic_data(
            eval_results, 
            quality_threshold, 
            confidence_threshold
        )
        
        print("Loading original data...")
        train_data = load_data_file(train_file)
        dev_data = load_data_file(dev_file)
        
        print(f"Original train samples: {len(train_data)}")
        print(f"Original dev samples: {len(dev_data)}")
        
        # Combine original and filtered synthetic data
        combined_data = train_data + filtered_synthetic
        
        print(f"Combined data: {len(combined_data)} samples")
        print(f"Original: {len(train_data)}, Synthetic: {len(filtered_synthetic)}")
        
        # Save combined data
        output_file = os.path.join(output_dir, "train_balanced_filtered.jsonl")
        print(f"Saving to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in combined_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved combined data to {output_file}")
        
        # Save statistics
        stats = {
            "original_train_samples": len(train_data),
            "original_dev_samples": len(dev_data),
            "filtered_synthetic_samples": len(filtered_synthetic),
            "combined_samples": len(combined_data),
            "quality_threshold": quality_threshold,
            "confidence_threshold": confidence_threshold
        }
        
        stats_file = os.path.join(output_dir, "filtering_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Saved statistics to {stats_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()