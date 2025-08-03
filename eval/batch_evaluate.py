#!/usr/bin/env python3
"""
Simple batch evaluation script for synthetic data quality.
Uses ThreadPoolExecutor for parallel processing.
"""

import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from evaluate_synthetic_quality import DialogueQualityEvaluator


def evaluate_batch_parallel(evaluator, data, max_workers=5):
    """Evaluate data in parallel using ThreadPoolExecutor."""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {}
        for i, sample in enumerate(data):
            future = executor.submit(
                evaluator.evaluate_single_sample,
                sample['dialogue_text'],
                sample['label']
            )
            future_to_sample[future] = (i, sample)
        
        # Process completed tasks with progress bar
        with tqdm(total=len(data), desc="Evaluating samples") as pbar:
            for future in as_completed(future_to_sample):
                i, sample = future_to_sample[future]
                try:
                    result = future.result()
                    result['sample_id'] = i
                    result['is_synthetic'] = sample['is_synthetic']
                    results.append(result)
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    # Add error result
                    error_result = {
                        'sample_id': i,
                        'is_synthetic': sample['is_synthetic'],
                        'quality_score': 0,
                        'confidence': 0.0,
                        'overall_rating': f"处理错误: {str(e)}"
                    }
                    results.append(error_result)
                
                pbar.update(1)
    
    # Sort results by sample_id to maintain order
    results.sort(key=lambda x: x['sample_id'])
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch evaluate synthetic dialogue quality")
    parser.add_argument("--api_key", required=True, help="DeepSeek API key")
    parser.add_argument("--data_file", default="output_synthetic_quality/synthetic_data.json", 
                       help="Path to extracted JSON data file")
    parser.add_argument("--sample_size", type=int, default=1000, 
                       help="Number of samples to evaluate")
    parser.add_argument("--output_file", default="output_synthetic_quality/batch_results.json", 
                       help="Output file path")
    parser.add_argument("--max_workers", type=int, default=500, 
                       help="Maximum number of parallel workers (DeepSeek has no rate limits)")
    parser.add_argument("--model", default="deepseek-chat", help="Model to use")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DialogueQualityEvaluator(args.api_key)
    
    # Load data
    print(f"Loading synthetic data from {args.data_file}...")
    data = evaluator.load_data(args.data_file, args.sample_size)
    print(f"Loaded {len(data)} synthetic samples for evaluation")
    
    # Evaluate in parallel
    print("Starting batch evaluation...")
    results = evaluate_batch_parallel(evaluator, data, args.max_workers)
    
    # Calculate statistics
    stats = evaluator.calculate_statistics(results)
    
    # Print summary
    evaluator.print_summary(stats)
    
    # Save results
    evaluator.save_results(results, stats, args.output_file)
    
    # Generate plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from plot_results import plot_quality_distribution, print_statistics
        
        print("\nGenerating plots...")
        print_statistics(results)
        
        # Create plot filename
        plot_file = args.output_file.replace('.json', '_distribution.png')
        plot_quality_distribution(results, plot_file)
        
    except ImportError:
        print("Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"Error generating plots: {e}")


if __name__ == "__main__":
    main()
