# Evaluation Framework

This directory contains comprehensive evaluation tools for CMCC-34 intent classification models and synthetic data quality assessment.

## Overview

The evaluation framework consists of two main components:

1. **Model Evaluation**: Evaluate fine-tuned models on intent classification tasks
2. **Synthetic Data Quality Assessment**: Evaluate the quality of synthetic dialogue samples

## Model Evaluation

### Main Evaluation Script (`evaluate.py`)

Evaluates fine-tuned models using system prompt optimization for CMCC-34 intent classification.

#### Usage

```bash
# Basic evaluation
python evaluate.py \
    --base_model_path THUDM/GLM-4-9B-0414 \
    --finetuned_path ../finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000 \
    --test_file ../data/cmcc-34/test.jsonl

# Quick evaluation
python evaluate.py \
    --quick \
    --samples 100 \
    --batch_size 1000 \
    --output_dir output_quick \
    --model_path ../finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000 \
    --test_file ../data/cmcc-34/test.jsonl
```

#### Command Line Options
- `--base_model_path`: Path to the base model (required)
- `--finetuned_path`: Path to the fine-tuned model (required)
- `--test_file`: Path to test data file (required)
- `--output_dir`: Output directory for results (default: "evaluation_output")
- `--use_4bit`: Use 4-bit quantization (default: True)

#### Output Files
- `results.json`: Detailed evaluation results
- `failed_predictions.json`: Samples where model failed to predict correctly
- `error_predictions.json`: Samples with parsing errors
- `confusion_matrix.json`: Confusion matrix data
- `detailed_analysis.json`: Per-category analysis
- `final_results.txt`: Human-readable summary

## Synthetic Data Quality Assessment

### Data Extraction (`extract_synthetic_data.py`)

Extract synthetic dialogue samples from balanced training data.

#### Usage

```bash
# Extract to JSON (default)
python extract_synthetic_data.py

# Extract to CSV
python extract_synthetic_data.py --format csv

# Custom input/output files
python extract_synthetic_data.py \
    --input_file ../aug/data/train_balanced.csv \
    --output_file my_synthetic_data.json \
    --format json
```

#### Command Line Options
- `--input_file`: Input CSV file path (default: `../aug/data/train_balanced.csv`)
- `--output_file`: Output file path (default: `synthetic_data.json`)
- `--format`: Output format - `json` or `csv` (default: `json`)

### Sequential Evaluation (`evaluate_synthetic_quality.py`)

Evaluate synthetic dialogue quality using DeepSeek API for small samples.

#### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run sequential evaluation
python evaluate_synthetic_quality.py --api_key YOUR_DEEPSEEK_API_KEY
```

#### Command Line Options
- `--api_key`: DeepSeek API key (required)
- `--data_file`: Path to extracted JSON data file (default: `output_synthetic_quality/synthetic_data.json`)
- `--sample_size`: Number of samples to evaluate (default: 10)
- `--output_file`: Output file path (default: `output_synthetic_quality/results.json`)

### Batch Evaluation (`batch_evaluate.py`)

Parallel batch evaluation for large samples using ThreadPoolExecutor.

#### Usage

```bash
# Run batch evaluation with parallel processing
python batch_evaluate.py --api_key YOUR_DEEPSEEK_API_KEY --sample_size 1000 --max_workers 500
```

#### Command Line Options
- `--api_key`: DeepSeek API key (required)
- `--data_file`: Path to extracted JSON data file (default: `output_synthetic_quality/synthetic_data.json`)
- `--sample_size`: Number of samples to evaluate (default: 1000)
- `--output_file`: Output file path (default: `output_synthetic_quality/batch_results.json`)
- `--max_workers`: Number of parallel threads (default: 500)

### Advanced Batch Evaluation (`evaluate_synthetic_batch.py`)

Enhanced batch evaluation with async processing and advanced features.

#### Usage

```bash
# Run advanced batch evaluation
python evaluate_synthetic_batch.py --api_key YOUR_DEEPSEEK_API_KEY --sample_size 1000 --max_workers 500
```

### Visualization (`plot_results.py`)

Generate comprehensive visualizations of evaluation results.

#### Usage

```bash
# Generate plots from existing results
python plot_results.py --results_file output_synthetic_quality/batch_results.json
```

#### Command Line Options
- `--results_file`: Path to evaluation results JSON file
- `--output_plot`: Output plot file path (default: `output_synthetic_quality/quality_distribution.png`)

## Evaluation Dimensions

### Synthetic Data Quality Assessment

The framework evaluates synthetic dialogue samples across three dimensions:

1. **Semantic Consistency** (40% weight)
   - Strict matching of labeled sub-intent
   - Intent boundary clarity (e.g., distinguishing "咨询" vs "办理")
   - Multi-intent mixing detection

2. **Context Completeness** (35% weight)
   - Must include user input and customer service response
   - Key element checks (business names, account info, time elements)
   - Multi-turn dialogue context coherence

3. **Language Naturalness** (25% weight)
   - Real oral features (pauses, reasonable grammar errors)
   - Domain terminology accuracy
   - Emotional expression rationality

### Scoring Standards
- **9-10 points**: Ideal samples requiring no modification
- **7-8 points**: Usable samples needing slight optimization
- **5-6 points**: Samples requiring moderate modification
- **3-4 points**: Samples with serious defects
- **1-2 points**: Invalid samples that should be discarded

## Performance Optimization

### Sequential vs Batch Processing
- **Sequential**: Best for small samples (10-50), simple debugging
- **Batch**: Optimized for large samples (1000+), uses parallel processing
- **Speed**: Batch processing with 500 workers can be 50x faster than sequential

### DeepSeek API Optimization
- **No Rate Limits**: DeepSeek doesn't constrain user rate limits
- **High Concurrency**: Can use 500+ parallel workers
- **Timeout Handling**: 120-second timeout to handle server delays
- **Retry Logic**: Exponential backoff for failed requests

## Business Intent Categories

The framework supports 34 business intent categories:

```
0: 咨询（含查询）业务规定
1: 办理取消
2: 咨询（含查询）业务资费
3: 咨询（含查询）营销活动信息
4: 咨询（含查询）办理方式
5: 投诉（含抱怨）业务使用问题
6: 咨询（含查询）账户信息
7: 办理开通
8: 咨询（含查询）业务订购信息查询
9: 投诉（含抱怨）不知情定制问题
10: 咨询（含查询）产品/业务功能
11: 咨询（含查询）用户资料
12: 投诉（含抱怨）费用问题
13: 投诉（含抱怨）业务办理问题
14: 投诉（含抱怨）服务问题
15: 办理变更
16: 咨询（含查询）服务渠道信息
17: 投诉（含抱怨）业务规定不满
18: 投诉（含抱怨）营销问题
19: 投诉（含抱怨）网络问题
20: 办理停复机
21: 投诉（含抱怨）信息安全问题
22: 办理重置/修改/补发
23: 咨询（含查询）使用方式
24: 咨询（含查询）号码状态
25: 咨询（含查询）工单处理结果
26: 办理打印/邮寄
27: 咨询（含查询）宽带覆盖范围
28: 办理移机/装机/拆机
29: 办理缴费
30: 办理下载/设置
31: 办理补换卡
32: 办理销户/重开
33: 咨询（含查询）电商货品信息
```

## Output Structure

### Model Evaluation Output
```
evaluation_output/
├── results.json              # Detailed evaluation results
├── failed_predictions.json   # Incorrect predictions
├── error_predictions.json    # Parsing errors
├── confusion_matrix.json     # Confusion matrix data
├── detailed_analysis.json    # Per-category analysis
└── final_results.txt        # Human-readable summary
```

### Synthetic Quality Assessment Output
```
output_synthetic_quality/
├── synthetic_data.json       # Extracted synthetic data
├── batch_results.json        # Evaluation results
├── quality_distribution.png  # Quality score plots
└── results.json             # Sequential evaluation results
```

## Dependencies

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers`: Model loading and inference
- `peft`: Parameter-efficient fine-tuning
- `torch`: PyTorch for model operations
- `scikit-learn`: Metrics calculation
- `tqdm`: Progress bars
- `matplotlib`: Visualization
- `requests`: API calls
- `numpy`: Numerical operations

## Notes

- Uses DeepSeek Chat API with temperature=0.1 for consistent evaluations
- Includes retry logic with exponential backoff for API calls
- No rate limiting needed (DeepSeek has no rate limits)
- Truncates long texts to 2000 characters for API efficiency
- Evaluates synthetic dialogue samples with detailed analysis
- Provides detailed JSON output with improvement suggestions
- Uses weighted scoring framework for quality assessment
- Automatic plotting of quality score distributions 