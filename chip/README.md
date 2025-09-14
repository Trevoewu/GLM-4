# Chip Data Conversion

This directory contains scripts for converting medical data to GLM-4 fine-tuning format.

## Files

- `data_conversion_script.py` - Main script to convert medical records to GLM-4 format
- `prompt.py` - Contains system and user prompts for medication prediction
- `候选药物列表.json` - List of candidate drugs for medication prediction

## Usage

### Convert Medical Data

```bash
cd chip/
python3 data_conversion_script.py
```

This will convert the medical data from:
- `../data/CDrugRed-A-v1/CDrugRed_train.jsonl` → `../data/CDrugRed-A-v1/train.json`
- `../data/CDrugRed-A-v1/CDrugRed_test-A.jsonl` → `../data/CDrugRed-A-v1/test.json`

### Output Format

The converted data follows GLM-4 conversation format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt for medication prediction..."
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

## Notes

- The script uses absolute paths for better reliability
- Output format is simplified to only include medication lists
- Large dataset files are ignored by git (see .gitignore)
