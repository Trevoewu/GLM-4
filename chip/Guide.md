# Complete Fine-tuning Guide for GLM-4-9B Medication Prediction

## Step 1: Prepare Your Environment

```bash
# Clone the GLM-4 repository
git clone https://github.com/THUDM/GLM-4.git
cd GLM-4/finetune

# Install dependencies
pip install -r requirements.txt
pip install swanlab  # Optional: for experiment tracking
```

## Step 2: Convert Your Data

1. **Prepare your original data**: Ensure your medical records are in the format you showed earlier.

2. **Run the conversion script**:
```python
# Place the data conversion script in the GLM-4/finetune directory
python data_conversion_script.py
```

3. **Verify the converted data format**:
```python
import json
with open('train.json', 'r', encoding='utf-8') as f:
    sample = json.load(f)[0]
    print(json.dumps(sample, ensure_ascii=False, indent=2))
```

## Step 3: Create Directory Structure

```bash
# Create data directory for your medication prediction task
mkdir -p data/medication_prediction/

# Move your converted data files
mv train.json data/medication_prediction/
mv val.json data/medication_prediction/
mv test.json data/medication_prediction/

# Copy and modify the configuration
cp configs/lora.yaml configs/medication_lora.yaml
# Edit configs/medication_lora.yaml with the provided configuration
```

## Step 4: Modify Configuration

Replace the content of `configs/medication_lora.yaml` with the configuration provided above, paying attention to:

- **Batch size**: Adjust based on your GPU memory
- **Learning rate**: Start with 1e-4, may need tuning
- **Max input length**: 8192 tokens for long medical records
- **LoRA parameters**: r=16, alpha=32 for good balance

## Step 5: Start Fine-tuning

### Single GPU (Recommended for testing):
```bash
python finetune.py data/medication_prediction/ THUDM/GLM-4-9B-0414 configs/medication_lora.yaml
```

### Multi-GPU (For faster training):
```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 finetune.py data/medication_prediction/ THUDM/GLM-4-9B-0414 configs/medication_lora.yaml
```

## Step 6: Monitor Training

- Check logs in `./logs` directory
- If using SwanLab, visit the dashboard URL shown in console
- Monitor key metrics:
  - Training loss (should decrease)
  - Validation loss (should decrease without overfitting)
  - Learning rate schedule

## Step 7: Evaluate Your Model

After training completes, test your model:

```python
from finetune_demo.inference import load_model_and_tokenizer

# Load your fine-tuned model
model_path = "./medication_prediction_model"  # Your output directory
model, tokenizer = load_model_and_tokenizer(model_path)

# Test with a sample medical record
test_prompt = """基于以下中文电子病历，推荐最佳的出院用药方案...
[Your medical record here]"""

# Generate prediction
inputs = tokenizer(test_prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)
```

## Step 8: Model Optimization Tips

### If you encounter memory issues:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use `fp16: true` for mixed precision training
- Consider using DeepSpeed ZeRO-2 configuration

### If training is too slow:
- Increase batch size if GPU memory allows
- Use multiple GPUs with torchrun
- Consider using smaller LoRA rank (r=8) for faster training

### If model performance is poor:
- Increase LoRA rank (r=32 or r=64)
- Adjust learning rate (try 5e-5 or 2e-4)
- Increase training epochs
- Ensure data quality and format correctness

## Step 9: Continue Training from Checkpoint

If training is interrupted, resume from the last checkpoint:
```bash
python finetune.py data/medication_prediction/ THUDM/GLM-4-9B-0414 configs/medication_lora.yaml yes
```

## Expected Results

After successful fine-tuning, your model should:
- Generate medication lists in proper JSON format
- Consider patient-specific factors (age, BMI, diagnoses)
- Provide clinical reasoning for drug selection
- Avoid obvious contraindications
- Maintain consistency with training examples

## Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce batch size, use gradient checkpointing
2. **Data format errors**: Verify JSON structure matches GLM-4 requirements
3. **Poor convergence**: Check learning rate, data quality, and model configuration
4. **Generation issues**: Adjust temperature, top_p, and max_new_tokens

**Hardware Requirements:**
- Minimum: 1x GPU with 24GB VRAM (e.g., RTX 4090, A5000)
- Recommended: 1x GPU with 48GB+ VRAM (e.g., A6000, H100)
- For full fine-tuning: Multiple GPUs with 80GB+ total VRAM