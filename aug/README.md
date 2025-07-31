# CMCC-34 Dataset LLM-based Data Augmentation

## 📋 Overview

This directory contains an improved **LLM-based data augmentation system** specifically designed to address severe class imbalance in the CMCC-34 dataset. The system uses GLM-4 model to generate high-quality synthetic samples for minority classes.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Start GLM-4 API Server on 8001 port
```bash
# Start the GLM-4 API server (runs on port 8001)
cd ../inference && python glm4v_server.py
```
> You should keep the server running until the augmentation is finished.

### 3. Run Data Augmentation

```bash
cd ../aug && python run_augmentation.py              # Run full augmentation
```

### 4. Generate Visualizations and Analysis

```bash
# Generate comprehensive analysis and visualizations
python visualization.py
```

### 5. Train the model on the balanced dataset

```bash
cd ../finetune && python finetune.py  ../aug/data THUDM/GLM-4-9B-0414  configs/balanced_qlora.yaml 
```

### 6. Evaluate the performance

```bash
python evaluate.py --model-path ../finetune/output/cmcc34_qlora_balanced_system_prompt/checkpoint-5000
```

## 📚 References

- **GLM-4 Model**: [THUDM/GLM-4-9B-0414](https://huggingface.co/THUDM/GLM-4-9B-0414)
- **CMCC-34 Dataset**: Telecom customer service intent classification
- **Evaluation Metrics**: Gini coefficient, F1-score, precision/recall
- **Technical Paper**: [LLM-based Data Augmentation for Imbalanced Text Classification]

---

💡 **Pro Tip**: Start with dry-run mode to verify configuration, then run full augmentation during off-peak hours due to computational requirements.
