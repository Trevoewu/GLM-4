# GLM-4 Fine-tuning for CMCC-34 Intent Classification

This repository contains a complete pipeline for fine-tuning GLM-4-9B on the CMCC-34 dataset for intent classification using QLoRA (Quantized Low-Rank Adaptation). The project includes data preparation, **LLM-baseddata augmentation**, model training, evaluation, and inference capabilities.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install torch transformers peft bitsandbytes accelerate
pip install scikit-learn matplotlib seaborn tqdm

# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Data Preparation & Augmentation

```bash
# Convert raw CSV data to GLM-4 format with system prompts
cd data/cmcc-34
python convert_data.py

# Run LLM-based data augmentation to address class imbalance
cd ../aug
python run_augmentation.py --dry-run    # Test configuration
python run_augmentation.py              # Run full augmentation

### 3. Model Training

```bash
# Train the model using QLoRA
cd finetune
python train_cmcc34_system_prompt.py
```

### 4. Model Evaluation

```bash
# Quick evaluation (100 samples)
cd evaluation
python evaluate.py --q

# Full evaluation
python evaluate.py --model-path ../finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000
```

### 5. Model Inference

```bash
# Interactive CLI
cd inference
python trans_cli_finetuned_demo.py

# Web interface
python trans_web_finetuned_demo.py

## 🆕 Data Augmentation Pipeline

### Problem Statement

The CMCC-34 dataset suffers from severe class imbalance:
- **Most frequent class**: 1,961 samples (Class 0: 咨询业务规定)
- **Least frequent classes**: 1 sample each (Class 32: 办理销户/重开, Class 33: 咨询电商货品信息)
- **Imbalance ratio**: 1961:1
- **Gini coefficient**: 0.651 (high inequality)


### Usage
```bash
# Start GLM-4 API server
cd inference
python glm4v_server.py

# Run augmentation
cd ../aug
python run_augmentation.py --dry-run    # Test configuration
python run_augmentation.py              # Run full augmentation

## 🔧 Configuration

### Training Configuration

The training uses QLoRA with the following key parameters:

- **Base Model**: GLM-4-9B-0414
- **Quantization**: 4-bit (QLoRA)
- **LoRA Rank**: 16
- **LoRA Alpha**: 64
- **Learning Rate**: 2e-4
- **Batch Size**: 4
- **Max Steps**: 5000

### Evaluation Configuration

- **Test Dataset**: CMCC-34 test set
- **Metrics**: Accuracy, F1-Macro, F1-Weighted
- **Output**: Failed predictions, confusion matrix, detailed analysis

## 📊 Model Performance

### Available Checkpoints

Multiple checkpoints are available in `finetune/output/cmcc34_qlora_system_prompt/`:

- `checkpoint-500/` - Early training
- `checkpoint-1000/` - 1000 steps
- `checkpoint-2000/` - 2000 steps
- `checkpoint-3000/` - 3000 steps
- `checkpoint-4000/` - 4000 steps
- `checkpoint-5000/` - Latest (recommended)

### Performance Metrics

The model achieves:
- **Accuracy**: ~85-90% (depending on checkpoint)
- **F1-Macro**: ~0.85-0.90
- **F1-Weighted**: ~0.85-0.90

## 🎯 Intent Classification

The model classifies user intents into 34 categories:

| Intent ID | Intent Name |
|-----------|-------------|
| 0 | 咨询（含查询）业务规定 |
| 1 | 办理取消 |
| 2 | 咨询（含查询）业务资费 |
| 3 | 咨询（含查询）营销活动信息 |
| 4 | 咨询（含查询）办理方式 |
| 5 | 投诉（含抱怨）业务使用问题 |
| 6 | 咨询（含查询）账户信息 |
| 7 | 办理开通 |
| 8 | 咨询（含查询）业务订购信息查询 |
| 9 | 投诉（含抱怨）不知情定制问题 |
| 10 | 咨询（含查询）产品/业务功能 |
| ... | ... |

## 🛠️ Usage Examples

### Command Line Evaluation

```bash
# Quick evaluation with 50 samples
python evaluate.py --quick --samples 50

# Full evaluation with custom model path
python evaluate.py --model-path ../finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000

# Custom batch size and output directory
python evaluate.py --batch-size 25 --output-dir my_evaluation_results
```

### Programmatic Usage

```python
from evaluation.evaluate import SystemPromptEvaluator

# Initialize evaluator
evaluator = SystemPromptEvaluator(
    base_model_path="THUDM/GLM-4-9B-0414",
    finetuned_path="finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000",
    test_file="finetune/data/cmcc-34/test.jsonl",
    output_dir="evaluation_results"
)

# Load model and evaluate
evaluator.load_model()
test_data = evaluator.load_test_data()
results = evaluator.evaluate_batch(test_data)

# Print results
evaluator.print_results(results)
evaluator.save_final_results(results)
```

### Interactive Inference

```bash
# Start interactive CLI
cd inference
python trans_cli_finetuned_demo.py

# Example conversation:
# User: 我想查询我的余额
# Assistant: 意图：6:咨询（含查询）账户信息

# GLM-4V Vision model
python trans_cli_vision_demo.py

# Batch inference
python trans_batch_demo.py
```

## 📈 Evaluation Results

The evaluation script generates comprehensive results:

### Output Files

- `failed_predictions.json` - Failed prediction details
- `error_predictions.json` - Wrong predictions analysis
- `confusion_matrix.png` - Visual confusion matrix
- `detailed_analysis.json` - Per-class performance metrics
- `system_prompt_evaluation_results.json` - Complete results

### Analysis Features

- **Failed Prediction Analysis**: Detailed error categorization
- **Confusion Matrix**: Visual representation of classification errors
- **Per-Class Performance**: Precision, recall, F1-score for each intent
- **Error Distribution**: Most confused intent pairs
- **Retry Logic**: Automatic handling of long inputs and memory errors

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python evaluate.py --batch-size 10
   
   # Use smaller checkpoint
   python evaluate.py --model-path checkpoint-2000
   ```

2. **Model Loading Errors**
   ```bash
   # Check model path
   ls finetune/output/cmcc34_qlora_system_prompt/
   
   # Verify base model
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('THUDM/GLM-4-9B-0414')"
   ```

3. **Data Loading Issues**
   ```bash
   # Regenerate dataset
   cd finetune/data/cmcc-34
   python regenerate_dataset.py
   ```

### Memory Requirements

- **Training**: ~24GB GPU memory (with QLoRA)
- **Evaluation**: ~10GB GPU memory
- **Inference**: ~10GB GPU memory
- **Data Augmentation**: ~15GB GPU memory

## 🚀 Advanced Usage

### Custom Training

```bash
# Modify training config
vim finetune/configs/cmcc34_qlora_system_prompt.yaml

# Train with custom parameters
python train_cmcc34_system_prompt.py --config custom_config.yaml
```

### Custom Evaluation

```bash
# Evaluate on custom test set
python evaluate.py --test-file custom_test.jsonl

# Compare multiple checkpoints
for checkpoint in 1000 2000 3000 4000 5000; do
    python evaluate.py --model-path checkpoint-$checkpoint --output-dir eval_$checkpoint
done
```

### Production Deployment

```python
# Load model for production
from inference.model_loader import load_finetuned_model

model, tokenizer = load_finetuned_model(
    base_model_path="THUDM/GLM-4-9B-0414",
    finetuned_path="finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000"
)

# Batch inference
def classify_intents(texts):
    results = []
    for text in texts:
        intent = model.generate_intent(text)
        results.append(intent)
    return results
```

## 📚 References

- [GLM-4 Paper](https://arxiv.org/abs/2401.09602)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [CMCC-34 Dataset](https://github.com/THUDM/GLM-4)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- THUDM for the GLM-4 model
- Microsoft for the QLoRA technique
- The open-source community for various tools and libraries