#!/usr/bin/env python3
"""
Filter synthetic data based on quality_score and confidence thresholds.
This script reads evaluation results containing synthetic data with quality metrics
and filters them according to specified thresholds.
"""

import json
import os
import argparse
from typing import List, Dict, Any
from pathlib import Path


class SyntheticDataFilter:
    """Filter synthetic data based on quality_score and confidence thresholds."""
    
    def __init__(self, quality_threshold: float = 7.0, confidence_threshold: float = 0.8):
        """
        Initialize the filter with quality and confidence thresholds.
        
        Args:
            quality_threshold: Minimum quality score (0-10 scale)
            confidence_threshold: Minimum confidence score (0-1 scale)
        """
        self.quality_threshold = quality_threshold
        self.confidence_threshold = confidence_threshold
    
    def load_evaluation_results(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load evaluation results from JSON file.
        
        Args:
            file_path: Path to the evaluation results JSON file
            
        Returns:
            List of evaluation results with quality scores and confidence
        """
        print(f"Loading evaluation results from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'detailed_results' in data:
            results = data['detailed_results']
            print(f"Loaded {len(results)} evaluation results")
            return results
        else:
            # If the file contains results directly
            print(f"Loaded {len(data)} evaluation results")
            return data
    
    def filter_data(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter data based on quality_score and confidence thresholds.
        
        Args:
            results: List of evaluation results
            
        Returns:
            List of filtered results that meet the thresholds
        """
        print(f"Filtering data with quality_threshold={self.quality_threshold}, confidence_threshold={self.confidence_threshold}")
        
        filtered_results = []
        total_samples = len(results)
        
        for result in results:
            quality_score = result.get('quality_score', 0)
            confidence = result.get('confidence', 0.0)
            
            # Check if both thresholds are met
            if quality_score >= self.quality_threshold and confidence >= self.confidence_threshold:
                filtered_results.append(result)
        
        print(f"Filtered {len(filtered_results)} out of {total_samples} samples")
        print(f"Filter rate: {len(filtered_results)/total_samples*100:.2f}%")
        
        return filtered_results
    
    def extract_synthetic_data(self, filtered_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract the original synthetic data from filtered evaluation results.
        
        Args:
            filtered_results: List of filtered evaluation results
            
        Returns:
            List of synthetic data samples
        """
        synthetic_data = []
        
        for result in filtered_results:
            # Extract the original synthetic data fields
            synthetic_sample = {
                'dialogue_text': result.get('original_text', ''),
                'label': result.get('label', ''),
                'label_numerical': result.get('label_numerical', ''),
                'is_synthetic': result.get('is_synthetic', True),
                'quality_score': result.get('quality_score', 0),
                'confidence': result.get('confidence', 0.0)
            }
            
            # Add dimension analysis if available
            if 'dimension_analysis' in result:
                synthetic_sample['dimension_analysis'] = result['dimension_analysis']
            
            # Add improvement suggestions if available
            if 'improvement_suggestions' in result:
                synthetic_sample['improvement_suggestions'] = result['improvement_suggestions']
            
            synthetic_data.append(synthetic_sample)
        
        return synthetic_data
    
    def save_filtered_data(self, filtered_data: List[Dict[str, Any]], output_dir: str, 
                          filename: str = "filtered_synthetic_data.json"):
        """
        Save filtered data to output directory.
        
        Args:
            filtered_data: List of filtered synthetic data
            output_dir: Output directory path
            filename: Output filename
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(filtered_data)} filtered samples to {output_path}")
        
        # Also save statistics
        stats = self.calculate_filtered_statistics(filtered_data)
        stats_path = os.path.join(output_dir, "filtered_statistics.json")
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Saved statistics to {stats_path}")
    
    def calculate_filtered_statistics(self, filtered_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for filtered data.
        
        Args:
            filtered_data: List of filtered synthetic data
            
        Returns:
            Dictionary containing statistics
        """
        if not filtered_data:
            return {"error": "No filtered data available"}
        
        quality_scores = [item.get('quality_score', 0) for item in filtered_data]
        confidence_scores = [item.get('confidence', 0.0) for item in filtered_data]
        
        # Count by label
        label_counts = {}
        for item in filtered_data:
            label = item.get('label', 'Unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
        
        stats = {
            "total_filtered_samples": len(filtered_data),
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "min_quality_score": min(quality_scores) if quality_scores else 0,
            "max_quality_score": max(quality_scores) if quality_scores else 0,
            "min_confidence": min(confidence_scores) if confidence_scores else 0,
            "max_confidence": max(confidence_scores) if confidence_scores else 0,
            "label_distribution": label_counts,
            "filter_thresholds": {
                "quality_threshold": self.quality_threshold,
                "confidence_threshold": self.confidence_threshold
            }
        }
        
        return stats
    
    def print_summary(self, original_count: int, filtered_count: int, stats: Dict[str, Any]):
        """
        Print a summary of the filtering results.
        
        Args:
            original_count: Number of original samples
            filtered_count: Number of filtered samples
            stats: Statistics dictionary
        """
        print("\n" + "="*60)
        print("SYNTHETIC DATA FILTERING SUMMARY")
        print("="*60)
        print(f"Original samples: {original_count}")
        print(f"Filtered samples: {filtered_count}")
        print(f"Filter rate: {filtered_count/original_count*100:.2f}%")
        print(f"Quality threshold: {self.quality_threshold}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Average quality score: {stats.get('avg_quality_score', 0):.2f}")
        print(f"Average confidence: {stats.get('avg_confidence', 0):.2f}")
        print(f"Quality score range: {stats.get('min_quality_score', 0):.1f} - {stats.get('max_quality_score', 0):.1f}")
        print(f"Confidence range: {stats.get('min_confidence', 0):.3f} - {stats.get('max_confidence', 0):.3f}")
        print("="*60)


def main():
    """Main function to run the synthetic data filtering."""
    parser = argparse.ArgumentParser(description="Filter synthetic data based on quality_score and confidence")
    parser.add_argument("--input_file", 
                       default="output_synthetic_quality/batch_results.json",
                       help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", 
                       default="output_filtered",
                       help="Output directory for filtered data")
    parser.add_argument("--quality_threshold", 
                       type=float, default=8.0,
                       help="Minimum quality score threshold (0-10)")
    parser.add_argument("--confidence_threshold", 
                       type=float, default=0.8,
                       help="Minimum confidence threshold (0-1)")
    parser.add_argument("--output_filename", 
                       default="filtered_synthetic_data.json",
                       help="Output filename for filtered data")
    
    args = parser.parse_args()
    
    # Create filter
    filter_obj = SyntheticDataFilter(
        quality_threshold=args.quality_threshold,
        confidence_threshold=args.confidence_threshold
    )
    
    # Load evaluation results
    try:
        results = filter_obj.load_evaluation_results(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {args.input_file}")
        return
    
    # Filter data
    filtered_results = filter_obj.filter_data(results)
    
    # Extract synthetic data from filtered results
    synthetic_data = filter_obj.extract_synthetic_data(filtered_results)
    
    # Save filtered data
    filter_obj.save_filtered_data(synthetic_data, args.output_dir, args.output_filename)
    
    # Calculate and print statistics
    stats = filter_obj.calculate_filtered_statistics(synthetic_data)
    filter_obj.print_summary(len(results), len(filtered_results), stats)


if __name__ == "__main__":
    main()