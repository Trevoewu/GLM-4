#!/usr/bin/env python3
"""
Simple script to extract synthetic data from train_balanced.csv
Follows KISS principle - Keep It Simple, Stupid
"""

import csv
import json
import argparse
from typing import List, Dict


def extract_synthetic_data(csv_file: str, output_file: str = None) -> List[Dict]:
    """Extract synthetic data from CSV file."""
    synthetic_data = []
    
    print(f"Reading from {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # Start from 2 to account for header
            # Check if this is synthetic data (synthetic column contains "True")
            if row.get('synthetic') == 'True':
                synthetic_sample = {
                    'row_number': row_num,
                    'dialogue_text': row.get('sentence_sep', ''),
                    'label': row.get('label_raw', ''),
                    'label_numerical': row.get('c_numerical', ''),
                    'is_synthetic': True
                }
                synthetic_data.append(synthetic_sample)
    
    print(f"Found {len(synthetic_data)} synthetic samples")
    return synthetic_data


def save_to_json(data: List[Dict], output_file: str):
    """Save data to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {output_file}")


def save_to_csv(data: List[Dict], output_file: str):
    """Save data to CSV file."""
    if not data:
        print("No data to save")
        return
    
    fieldnames = ['row_number', 'dialogue_text', 'label', 'label_numerical', 'is_synthetic']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in data:
            writer.writerow(sample)
    
    print(f"Saved {len(data)} samples to {output_file}")


def print_summary(data: List[Dict]):
    """Print summary statistics."""
    if not data:
        print("No synthetic data found")
        return
    
    # Count by label
    label_counts = {}
    for sample in data:
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n" + "="*50)
    print("SYNTHETIC DATA EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total synthetic samples: {len(data)}")
    print(f"Unique labels: {len(label_counts)}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} samples")
    print("="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract synthetic data from train_balanced.csv")
    parser.add_argument("--input_file", default="../aug/data/train_balanced.csv", 
                       help="Input CSV file path")
    parser.add_argument("--output_file", default="output_synthetic_quality/synthetic_data.json", 
                       help="Output file path")
    parser.add_argument("--format", choices=['json', 'csv'], default='json',
                       help="Output format")
    
    args = parser.parse_args()
    
    # Extract synthetic data
    synthetic_data = extract_synthetic_data(args.input_file)
    
    if not synthetic_data:
        print("No synthetic data found in the file")
        return
    
    # Print summary
    print_summary(synthetic_data)
    
    # Save to file
    if args.format == 'json':
        save_to_json(synthetic_data, args.output_file)
    else:
        # Change extension to .csv if needed
        if not args.output_file.endswith('.csv'):
            args.output_file = args.output_file.replace('.json', '.csv')
        save_to_csv(synthetic_data, args.output_file)


if __name__ == "__main__":
    main() 