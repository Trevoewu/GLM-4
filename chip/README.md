# Chip Data Conversion

This directory contains scripts for converting medical data to GLM-4 fine-tuning format for medication prediction tasks.

## Files

- `data_conversion_script.py` - Main script to convert medical records to GLM-4 conversation format
- `prompt.py` - Contains system and user prompts for medication prediction
- `候选药物列表.json` - List of 651 candidate drugs for medication prediction
- `README.md` - This documentation file

## Usage

### 1. Convert Medical Data for Fine-tuning

```bash
cd chip/
python3 data_conversion_script.py
```

This will convert the medical data from:
- `../data/CDrugRed-A-v1/CDrugRed_train.jsonl` → `../data/CDrugRed-A-v1/train.json` (3,602 records)
- `../data/CDrugRed-A-v1/CDrugRed_test-A.jsonl` → `../data/CDrugRed-A-v1/test.json` (570 records)

### 2. Fine-tune GLM-4 Model

```bash
cd ../finetune
python finetune.py ../data/CDrugRed-A-v1 THUDM/GLM-4-9B-0414 configs/medication_lora.yaml
```

You can modify the training parameters in `configs/medication_lora.yaml` to adjust:
- Learning rate
- Batch size
- Number of epochs
- LoRA parameters

## Output Format

The converted data follows GLM-4 conversation format with simplified medication list output:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Medical AI assistant prompt..."
    },
    {
      "role": "user", 
      "content": "Patient medical record information..."
    },
    {
      "role": "assistant",
      "content": "{\n  \"出院带药列表\": [\"药物1\", \"药物2\", \"...\"]\n}"
    }
  ]
}
```

## Requirements

- Python 3.6+
- JSON data files in the specified directory structure
- GLM-4 model files for fine-tuning

## Notes

- The script uses absolute paths for better reliability
- Output format is simplified to only include medication lists
- Large dataset files are ignored by git (see .gitignore)
- The model is trained to predict discharge medications from Chinese medical records