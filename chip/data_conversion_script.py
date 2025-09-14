import json
from typing import List, Dict, Any
from pathlib import Path
from prompt import SYSTEM_PROMPT, USER_PROMPT

def convert_medical_data_for_glm4(input_data: List[Dict], output_file: str):
    """
    Convert medical records to GLM-4-9B fine-tuning format for medication prediction
    
    Args:
        input_data: List of medical record dictionaries
        output_file: Path to save the converted data
    """
    
    # Use system prompt from prompt.py
    system_prompt = SYSTEM_PROMPT.strip()

    converted_data = []
    
    for record in input_data:
        # Extract patient information and create user prompt
        user_prompt = create_user_prompt(record)
        
        # Create assistant response (expected medications)
        assistant_response = create_assistant_response(record)
        
        # Create the conversation format required by GLM-4-9B
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user", 
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ]
        }
        
        converted_data.append(conversation)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(converted_data)} records and saved to {output_file}")

def create_user_prompt(record: Dict) -> str:
    """Create user prompt from medical record using template from prompt.py"""
    
    # Calculate age if possible
    age_info = ""
    if "出生日期" in record and "就诊时间" in record:
        try:
            birth_year = int(record["出生日期"].split("-")[0])
            visit_year = int(record["就诊时间"].split("-")[0])
            age = visit_year - birth_year
            age_info = f"{age}岁"
        except:
            age_info = "未知"
    else:
        age_info = "未知"
    
    # Format the user prompt using the template from prompt.py
    # First replace the template variables, then handle the JSON example
    prompt = USER_PROMPT.replace('{患者序号}', str(record.get('患者序号', '未知')))
    prompt = prompt.replace('{性别}', str(record.get('性别', '未知')))
    prompt = prompt.replace('{出生日期}', str(record.get('出生日期', '未知')))
    prompt = prompt.replace('{就诊时间}', str(record.get('就诊时间', '未知')))
    prompt = prompt.replace('{BMI}', str(record.get('BMI', '未知')))
    prompt = prompt.replace('{既往史}', str(record.get('既往史', '未提供')))
    prompt = prompt.replace('{主诉}', str(record.get('主诉', '未提供')))
    prompt = prompt.replace('{现病史}', str(record.get('现病史', '未提供')))
    prompt = prompt.replace('{入院情况}', str(record.get('入院情况', '未提供')))
    prompt = prompt.replace('{诊疗过程描述}', str(record.get('诊疗过程描述', '未提供')))
    prompt = prompt.replace('{出院诊断}', str(record.get('出院诊断', [])))
    
    # Replace the age placeholder with calculated age
    prompt = prompt.replace("[Calculate from 出生日期 and 就诊时间]", age_info)
    
    return prompt

def create_assistant_response(record: Dict) -> str:
    """Create assistant response with expected medications only"""
    
    medications = record.get("出院带药列表", [])
    
    # Create simple response with only the medication list
    response = {
        "出院带药列表": medications
    }
    
    return json.dumps(response, ensure_ascii=False, indent=2)

def split_dataset(input_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataset into train, validation, and test sets"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Shuffle data
    import random
    random.shuffle(data)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Save splits
    base_path = Path(input_file).parent
    
    with open(base_path / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(base_path / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
        
    with open(base_path / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line.strip()))
    return data

# Example usage
if __name__ == "__main__":
    # Convert available datasets using absolute paths
    datasets = [
        ("/data/long/glm4/data/CDrugRed-A-v1/CDrugRed_train.jsonl", "/data/long/glm4/data/CDrugRed-A-v1/train.json"),
        ("/data/long/glm4/data/CDrugRed-A-v1/CDrugRed_test-A.jsonl", "/data/long/glm4/data/CDrugRed-A-v1/test.json")
    ]
    
    for input_file, output_file in datasets:
        print(f"Converting {input_file}...")
        original_data = load_jsonl_data(input_file)
        convert_medical_data_for_glm4(original_data, output_file)
    
    print("Data conversion completed!")
    print("Files created:")
    print("- /data/long/glm4/data/CDrugRed-A-v1/train.json")
    print("- /data/long/glm4/data/CDrugRed-A-v1/test.json")