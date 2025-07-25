#!/usr/bin/env python3
"""
Training script for CMCC-34 intent classification with system prompt optimization.
This script uses the new dataset format with system prompts to reduce token waste.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main training function."""
    print("Starting CMCC-34 training with system prompt optimization...")
    
    # Configuration
    data_dir = "../data/cmcc-34"
    model_dir = "THUDM/GLM-4-9B-0414"
    config_file = "configs/cmcc34_qlora_system_prompt.yaml"
    
    # Check if dataset exists
    dataset_files = ['train.jsonl', 'dev.jsonl', 'test.jsonl']
    missing_files = []
    
    for file in dataset_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Error: Missing dataset files: {file_path}")
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing dataset files: {missing_files}")
        print("Please run the data conversion script first:")
        print("  cd data/cmcc-34")
        print("  python regenerate_dataset.py")
        sys.exit(1)
    
    # Check if config file exists, create if not
    if not os.path.exists(config_file):
        print(f"Creating optimized config file: {config_file}")
        create_optimized_config(config_file)
    
    # Run training
    print(f"\nStarting training with:")
    print(f"  Data directory: {data_dir}")
    print(f"  Model: {model_dir}")
    print(f"  Config: {config_file}")
    
    cmd = [
        "python", "finetune.py",
        data_dir,
        model_dir,
        config_file,
        "yes"  # auto resume from checkpoint
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        sys.exit(1)

def create_optimized_config(config_file):
    """Create an optimized config file for system prompt training."""
    config_content = """data_config:
  train_file: train.jsonl
  val_file: dev.jsonl
  test_file: test.jsonl
  num_proc: 4

combine: True
freezeV: True
max_input_length: 2048  # Increased for system prompt
max_output_length: 256  # Reduced since we only need intent output
# swanlab: "local"  # set to local if don`t use cloud

training_args:
  # see `transformers.Seq2SeqTrainingArguments`
  output_dir: ./output/cmcc34_qlora_system_prompt
  max_steps: 5000
  # needed to be fit for the dataset
  learning_rate: 2e-4
  # settings for data loading
  per_device_train_batch_size: 4
  dataloader_num_workers: 4
  remove_unused_columns: false
  # settings for saving checkpoints
  save_strategy: steps
  save_steps: 500
  # settings for logging
  log_level: info
  logging_strategy: steps
  logging_steps: 50
  run_name: "glm4-cmcc34-qlora-system-prompt"
  # settings for evaluation
  per_device_eval_batch_size: 4
  eval_strategy: "steps"
  eval_steps: 500
  # settings for optimizer
  # adam_epsilon: 1e-6
  # uncomment the following line to detect nan or inf values
  # debug: underflow_overflow
  predict_with_generate: true
  # QLoRA specific settings
  bf16: true
  gradient_checkpointing: true
  # see `transformers.GenerationConfig`
  generation_config:
    max_new_tokens: 128  # Reduced since we only need intent output
  # set your absolute deepspeed path here
  # deepspeed: configs/ds_zero_3.json
  # deepspeed: configs/ds_zero_2.json

peft_config:
  peft_type: LORA
  task_type: CAUSAL_LM
  r: 16
  lora_alpha: 64
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# QLoRA specific quantization config
quantization_config:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
"""
    
    # Create configs directory if it doesn't exist
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"Created optimized config file: {config_file}")

if __name__ == "__main__":
    main()