# CMCC-34 Dataset LLM-based Data Augmentation

## 📋 Overview

This directory contains an improved **LLM-based data augmentation system** specifically designed to address severe class imbalance in the CMCC-34 dataset. The system uses GLM-4 model to generate high-quality synthetic samples for minority classes.

## 🎯 Problem Statement

The CMCC-34 dataset suffers from severe class imbalance:
- **Most frequent class**: 1,961 samples (Class 0: 咨询业务规定)
- **Least frequent classes**: 1 sample each (Class 32: 办理销户/重开, Class 33: 咨询电商货品信息)
- **Imbalance ratio**: 1961:1
- **Gini coefficient**: 0.651 (high inequality)

## 🚀 Key Features

### ✨ Improved GLM-4 Integration
- **Enhanced API Server**: Custom GLM-4 API compatible with OpenAI format
- **Robust Error Handling**: Automatic retry mechanism with exponential backoff
- **Batch Processing**: Generate 10+ samples per API call (vs 5 previously)
- **Smart Targeting**: Class-specific sample targets based on severity

### 📊 Advanced Configuration
- **Ultra-minority Classes**: Target 300 samples for classes with <10 samples
- **Minority Classes**: Target 200 samples for classes with <50 samples
- **Adaptive Thresholds**: Dynamic minority class identification
- **Quality Control**: Length, similarity, and keyword validation

### 🎨 Professional Output Organization
- **Structured Output**: All results organized in `output/` directory
- **Visual Analytics**: Comprehensive comparison plots and distributions
- **Detailed Reports**: Machine-readable summaries and human-readable analysis

## 📁 Directory Structure

```
aug/
├── 📊 Core Scripts
│   ├── run_aug.py                     # Main augmentation runner
│   ├── data_augmentation.py           # Improved GLM-4 augmentation logic
│   └── api_server.py                  # Custom GLM-4 API server
├── ⚙️ Configuration
│   └── augment_config.yaml            # Enhanced configuration with class-specific targets
├── 📈 Analysis & Visualization
│   ├── plot_final_comparison.py       # Three-way comparison (Original/GLM-4/Enhanced)
│   ├── plot_balanced_comparison.py    # GLM-4 vs Original comparison
│   ├── plot_simple_distribution.py    # Distribution analysis
│   └── generate_report.py             # Comprehensive reporting
├── 📁 Output (Generated)
│   ├── plots/                         # All visualization outputs
│   │   ├── final_three_way_comparison.png
│   │   ├── balanced_vs_original_comparison.png
│   │   └── cmcc34_class_distribution.png
│   ├── train_enhanced.csv             # GLM-4 augmented dataset
│   ├── data_augmentation_report.txt   # Detailed analysis report
│   └── augmentation_summary.json      # Machine-readable summary
└── 📚 Documentation
    ├── README.md                      # This file
    └── README_augmentation.md          # Technical documentation
```

## 🚀 Quick Start

### 1. Start GLM-4 API Server
```bash
# Start the GLM-4 API server (runs on port 8001)
python src/scripts/api_server.py
```

### 2. Run Data Augmentation

```bash
# Option 1: Use the convenient launcher (recommended)
python run_augmentation.py --dry-run    # Test configuration
python run_augmentation.py              # Run full augmentation

# Option 2: Run directly from src (ensure PYTHONPATH is set)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/scripts/run_aug.py --dry-run
python src/scripts/run_aug.py

# Generate comprehensive analysis
python src/utils/generate_report.py
python src/plotting/plot_final_comparison.py
```

### 3. View Results

```bash
# Check output structure
ls -la outputs/
ls -la outputs/plots/

# View analysis report
cat outputs/reports/data_augmentation_report.txt
```

## 📊 Expected Results

### 🎯 Augmentation Targets
| Class Type | Classes | Original Samples | Target Samples | Improvement |
|------------|---------|------------------|----------------|-------------|
| **Ultra-minority** | 32, 33, 29, 30, 31 | 1-5 each | 300 each | 60-300x |
| **Minority** | 20, 22-28 | 15-49 each | 200 each | 4-13x |

### 📈 Expected Improvements
- **Dataset Growth**: ~3,000+ new synthetic samples
- **Gini Coefficient**: 0.651 → ~0.400 (significant improvement)
- **Imbalance Ratio**: 1961:1 → ~10:1 (dramatic reduction)
- **Model Performance**: Better F1-scores on minority classes

## ⚙️ Configuration

### Key Parameters in `augment_config.yaml`:

```yaml
augmentation:
  target_samples_per_class: 200    # Default target per class
  batch_size: 10                   # Samples per API call
  max_retries: 5                   # API failure tolerance
  delay_between_calls: 0.5         # API rate limiting

class_specific_strategies:
  ultra_minority:
    classes: [32, 33, 29, 30, 31]
    target_samples: 300            # Aggressive augmentation
  minority:
    classes: [20, 22, 23, 24, 25, 26, 27, 28]
    target_samples: 200            # Moderate augmentation
```

## 🔧 Technical Improvements

### vs. Original Implementation
| Aspect | Original | Improved |
|--------|----------|----------|
| **Batch Size** | 5 samples | 10 samples |
| **Retry Logic** | Basic | Exponential backoff |
| **Class Targeting** | Fixed threshold | Class-specific targets |
| **Error Handling** | Minimal | Comprehensive |
| **Output Organization** | Scattered | Structured in `output/` |
| **Documentation** | Basic | Comprehensive |

### API Compatibility
- **Endpoint**: `http://localhost:8001/v1/chat/completions`
- **Format**: OpenAI-compatible JSON
- **Model**: GLM-4-9B-0414 (local deployment)
- **Features**: Streaming, temperature control, token limits

## 📊 Analysis Tools

### Visualization Scripts
1. **`plot_final_comparison.py`**: Three-way comparison showing Original → GLM-4 → Enhanced results
2. **`plot_balanced_comparison.py`**: Detailed GLM-4 vs Original analysis
3. **`plot_simple_distribution.py`**: Clean distribution visualization

### Reporting Tools
1. **`generate_report.py`**: Comprehensive statistical analysis
2. **`augmentation_summary.json`**: Machine-readable metrics
3. **Automated quality checks**: Length, similarity, keyword validation

## 🚨 Troubleshooting

### Common Issues

1. **GLM-4 Server Not Starting**
   ```bash
   # Check if port 8001 is free
   netstat -tulpn | grep 8001
   
   # Kill existing process if needed
   pkill -f api_server
   ```

2. **Out of Memory**
   ```bash
   # Monitor GPU memory
   nvidia-smi
   
   # Reduce batch_size in config
   batch_size: 5  # Instead of 10
   ```

3. **API Connection Failed**
   ```bash
   # Test API health
   curl http://localhost:8001/health
   
   # Check server logs
   tail -f api_server.log
   ```

## 📈 Performance Monitoring

### Success Metrics
- **Generated Samples**: Target 2,000+ synthetic samples
- **Success Rate**: >80% API calls successful
- **Quality Score**: Generated samples pass validation
- **Balance Improvement**: Gini coefficient reduction >0.2

### Progress Tracking
```bash
# Monitor generation progress
tail -f augmentation.log

# Check current dataset size
wc -l output/train_enhanced.csv

# Analyze class distribution
python -c "
import pandas as pd
df = pd.read_csv('output/train_enhanced.csv')
print(df['c_numerical'].value_counts().sort_index())
"
```

## 🎯 Next Steps

1. **Train Model**: Use `output/train_enhanced.csv` for model training
2. **Evaluate Performance**: Compare F1-scores on minority classes
3. **Iterate**: Adjust targets based on model performance
4. **Scale**: Apply to other imbalanced datasets

## 📚 References

- **GLM-4 Model**: [THUDM/GLM-4-9B-0414](https://huggingface.co/THUDM/GLM-4-9B-0414)
- **CMCC-34 Dataset**: Telecom customer service intent classification
- **Evaluation Metrics**: Gini coefficient, F1-score, precision/recall
- **Technical Paper**: [LLM-based Data Augmentation for Imbalanced Text Classification]

---

💡 **Pro Tip**: Start with dry-run mode to verify configuration, then run full augmentation during off-peak hours due to computational requirements.
