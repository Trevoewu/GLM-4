#!/usr/bin/env python3
"""
Script to regenerate CMCC-34 dataset with system prompt optimization.
This script converts the original CSV files to the new JSONL format using system prompts.
"""

import os
import sys
from convert_data import convert_to_glm4_format

def main():
    """Main function to regenerate the dataset."""
    print("Regenerating CMCC-34 dataset with system prompt optimization...")
    
    # Check if CSV files exist
    csv_files = ['train_new.csv', 'dev_new.csv', 'test_new.csv']
    missing_files = [f for f in csv_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing CSV files: {missing_files}")
        print("Please ensure the following files exist in the current directory:")
        for f in csv_files:
            print(f"  - {f}")
        sys.exit(1)
    
    # Convert datasets
    print("\nConverting training set...")
    convert_to_glm4_format('train_new.csv', 'train.jsonl')
    
    print("\nConverting validation set...")
    convert_to_glm4_format('dev_new.csv', 'dev.jsonl')
    
    print("\nConverting test set...")
    convert_to_glm4_format('test_new.csv', 'test.jsonl')
    
    print("\nDataset regeneration completed!")
    print("Generated files:")
    print("  - train.jsonl")
    print("  - dev.jsonl") 
    print("  - test.jsonl")
    
    # Verify file sizes
    print("\nFile sizes:")
    for filename in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"  - {filename}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()