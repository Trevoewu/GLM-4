# Chip Data Conversion

This directory contains scripts for converting medical data to GLM-4 fine-tuning format.

## Files

- `data_conversion_script.py` - Main script to convert medical records to GLM-4 format
- `prompt.py` - Contains system and user prompts for medication prediction
- `候选药物列表.json` - List of candidate drugs for medication prediction

## Usage

### 1. Convert Medical Data for finetune

```bash
cd chip/
python3 data_conversion_script.py
```

This will convert the medical data from:
- `../data/CDrugRed-A-v1/CDrugRed_train.jsonl` → `../data/CDrugRed-A-v1/train.json`
- `../data/CDrugRed-A-v1/CDrugRed_test-A.jsonl` → `../data/CDrugRed-A-v1/test.json`

### 2. Finetune 

```bash
cd finetune
python finetune.py ../data/CDrugRed-A-v1 THUDM/GLM-4-9B-0414 configs/medication_lora.yaml
```
You could modify the training parameters in configs/medication_lora.yaml