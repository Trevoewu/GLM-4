#!/usr/bin/env python3
"""
Training script for CMCC-34 intent classification with system prompt optimization.
This script uses the new dataset format with system prompts to reduce token waste.
"""

import os
import sys
import subprocess
from pathlib import Path


import torch.serialization
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])


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
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
