"""
Batch evaluation script for the fine-tuned GLM-4 model on CMCC-34 test dataset.
This script can handle large datasets efficiently with progress tracking and checkpointing.
"""

import json
import re
import time
import torch
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM


class BatchEvaluator:
    """Batch evaluator for large-scale evaluation."""
    
    def __init__(self, base_model_path: str, finetuned_path: str, test_file: str, 
                 output_dir: str = "evaluation_output", use_4bit: bool = True):
        self.base_model_path = base_model_path
        self.finetuned_path = finetuned_path
        self.test_file = test_file
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Business type mapping
        self.business_types = {
            0: "咨询（含查询）业务规定", 1: "办理取消", 2: "咨询（含查询）业务资费", 3: "咨询（含查询）营销活动信息",
            4: "咨询（含查询）办理方式", 5: "投诉（含抱怨）业务使用问题", 6: "咨询（含查询）账户信息", 7: "办理开通",
            8: "咨询（含查询）业务订购信息查询", 9: "投诉（含抱怨）不知情定制问题", 10: "咨询（含查询）产品/业务功能",
            11: "咨询（含查询）用户资料", 12: "投诉（含抱怨）费用问题", 13: "投诉（含抱怨）业务办理问题",
            14: "投诉（含抱怨）服务问题", 15: "办理变更", 16: "咨询（含查询）服务渠道信息",
            17: "投诉（含抱怨）业务规定不满", 18: "投诉（含抱怨）营销问题", 19: "投诉（含抱怨）网络问题",
            20: "办理停复机", 21: "投诉（含抱怨）信息安全问题", 22: "办理重置/修改/补发",
            23: "咨询（含查询）使用方式", 24: "咨询（含查询）号码状态", 25: "咨询（含查询）工单处理结果",
            26: "办理打印/邮寄", 27: "咨询（含查询）宽带覆盖范围", 28: "办理移机/装机/拆机", 29: "办理缴费",
            30: "办理下载/设置", 31: "办理补换卡", 32: "办理销户/重开", 33: "咨询（含查询）电商货品信息"
        }
        
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the fine-tuned model."""
        print(f"Loading model: {self.finetuned_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        
        model_kwargs = {
            "use_cache": False,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_path, **model_kwargs)
        self.model = PeftModelForCausalLM.from_pretrained(self.model, self.finetuned_path)
        
        if self.use_4bit:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def load_test_data(self) -> List[Dict]:
        """Load all test data."""
        print(f"Loading test data from: {self.test_file}")
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} test samples")
        return data
    
    def extract_intent(self, response: str) -> Optional[int]:
        """Extract intent ID from response."""
        patterns = [
            r'意图[：:]\s*(\d+)[：:]\s*([^，。\n]+)',
            r'意图[：:]\s*(\d+)',
            r'(\d+)[：:]\s*([^，。\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    intent_id = int(match.group(1))
                    if intent_id in self.business_types:
                        return intent_id
                except ValueError:
                    continue
        
        # Fallback: find any number
        numbers = re.findall(r'\d+', response)
        for num in numbers:
            try:
                intent_id = int(num)
                if intent_id in self.business_types:
                    return intent_id
            except ValueError:
                continue
        
        return None
    
    def generate_response(self, user_content: str) -> str:
        """Generate response for a user message with system prompt."""
        # System prompt - same as used in training
        system_prompt = """你是客服意图识别专家。分析对话内容，判断用户最终意图。

对话格式：多个说话轮次用[SEP]分隔，通常以"您好请讲"开始。

判断标准：
- 关注用户最终目标，不是中间过程
- 结合关键词和上下文综合判断

业务类型列表：
0:咨询（含查询）业务规定 1:办理取消 2:咨询（含查询）业务资费 3:咨询（含查询）营销活动信息 4:咨询（含查询）办理方式
5:投诉（含抱怨）业务使用问题 6:咨询（含查询）账户信息 7:办理开通 8:咨询（含查询）业务订购信息查询 9:投诉（含抱怨）不知情定制问题
10:咨询（含查询）产品/业务功能 11:咨询（含查询）用户资料 12:投诉（含抱怨）费用问题 13:投诉（含抱怨）业务办理问题 14:投诉（含抱怨）服务问题
15:办理变更 16:咨询（含查询）服务渠道信息 17:投诉（含抱怨）业务规定不满 18:投诉（含抱怨）营销问题 19:投诉（含抱怨）网络问题
20:办理停复机 21:投诉（含抱怨）信息安全问题 22:办理重置/修改/补发 23:咨询（含查询）使用方式 24:咨询（含查询）号码状态
25:咨询（含查询）工单处理结果 26:办理打印/邮寄 27:咨询（含查询）宽带覆盖范围 28:办理移机/装机/拆机 29:办理缴费
30:办理下载/设置 31:办理补换卡 32:办理销户/重开 33:咨询（含查询）电商货品信息

输出格式：
意图：[选择最合适的意图]"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.8,
                temperature=0.6,
                repetition_penalty=1.2,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_ids = outputs[:, model_inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response.strip()
    
    def save_checkpoint(self, results: Dict, batch_num: int):
        """Save evaluation checkpoint."""
        checkpoint_file = os.path.join(self.output_dir, f"checkpoint_batch_{batch_num}.json")
        
        # Prepare results for saving
        save_results = results.copy()
        if 'classification_report' in save_results:
            # Convert numpy types to native Python types
            for class_name, metrics in save_results['classification_report'].items():
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, np.float64):
                            metrics[key] = float(value)
                        elif isinstance(value, np.int64):
                            metrics[key] = int(value)
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self, batch_num: int) -> Optional[Dict]:
        """Load evaluation checkpoint."""
        checkpoint_file = os.path.join(self.output_dir, f"checkpoint_batch_{batch_num}.json")
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    
    def evaluate_batch(self, test_data: List[Dict], batch_size: int = 100, 
                      start_batch: int = 0) -> Dict:
        """Evaluate data in batches with checkpointing."""
        print(f"Starting batch evaluation with batch_size={batch_size}")
        print(f"Total samples: {len(test_data)}")
        
        total_batches = (len(test_data) + batch_size - 1) // batch_size
        all_predictions = []
        all_ground_truth = []
        failed_samples = []
        
        start_time = time.time()
        
        for batch_num in range(start_batch, total_batches):
            print(f"\nProcessing batch {batch_num + 1}/{total_batches}")
            
            # Check if checkpoint exists
            checkpoint = self.load_checkpoint(batch_num)
            if checkpoint:
                print(f"Loading checkpoint for batch {batch_num}")
                all_predictions.extend(checkpoint['predictions'])
                all_ground_truth.extend(checkpoint['ground_truth'])
                failed_samples.extend(checkpoint.get('failed_samples', []))
                continue
            
            # Process current batch
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(test_data))
            batch_data = test_data[start_idx:end_idx]
            
            batch_predictions = []
            batch_ground_truth = []
            batch_failed = []
            
            for i, sample in enumerate(tqdm(batch_data, desc=f"Batch {batch_num + 1}")):
                # Handle new format with system prompt
                if len(sample["messages"]) == 3:  # system, user, assistant
                    user_message = sample["messages"][1]["content"]  # user message
                    assistant_message = sample["messages"][2]["content"]  # assistant message
                else:  # fallback for old format
                    user_message = sample["messages"][0]["content"]
                    assistant_message = sample["messages"][1]["content"]
                
                # Extract ground truth
                gt_intent = self.extract_intent(assistant_message)
                
                # Generate prediction
                try:
                    pred_response = self.generate_response(user_message)
                    pred_intent = self.extract_intent(pred_response)
                except Exception as e:
                    print(f"Error on sample {start_idx + i}: {e}")
                    pred_intent = None
                    batch_failed.append({
                        'index': start_idx + i,
                        'error': str(e),
                        'sample': sample
                    })
                
                if pred_intent is not None and gt_intent is not None:
                    batch_predictions.append(pred_intent)
                    batch_ground_truth.append(gt_intent)
                else:
                    batch_failed.append({
                        'index': start_idx + i,
                        'predicted': pred_intent,
                        'ground_truth': gt_intent,
                        'sample': sample
                    })
            
            # Save batch checkpoint
            batch_results = {
                'predictions': batch_predictions,
                'ground_truth': batch_ground_truth,
                'failed_samples': batch_failed,
                'batch_num': batch_num,
                'batch_size': len(batch_data)
            }
            self.save_checkpoint(batch_results, batch_num)
            
            # Accumulate results
            all_predictions.extend(batch_predictions)
            all_ground_truth.extend(batch_ground_truth)
            failed_samples.extend(batch_failed)
            
            # Print batch summary
            batch_accuracy = accuracy_score(batch_ground_truth, batch_predictions) if batch_predictions else 0
            print(f"Batch {batch_num + 1} - Accuracy: {batch_accuracy:.4f}, "
                  f"Successful: {len(batch_predictions)}, Failed: {len(batch_failed)}")
        
        evaluation_time = time.time() - start_time
        
        # Calculate final metrics
        if len(all_predictions) > 0:
            accuracy = accuracy_score(all_ground_truth, all_predictions)
            f1_macro = f1_score(all_ground_truth, all_predictions, average='macro')
            f1_weighted = f1_score(all_ground_truth, all_predictions, average='weighted')
            
            # Classification report
            class_names = [self.business_types.get(i, f"Class_{i}") for i in sorted(set(all_ground_truth + all_predictions))]
            report = classification_report(all_ground_truth, all_predictions, target_names=class_names, output_dict=True)
            
            results = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'total_samples': len(test_data),
                'successful_predictions': len(all_predictions),
                'failed_predictions': len(failed_samples),
                'evaluation_time': evaluation_time,
                'avg_time_per_sample': evaluation_time / len(test_data),
                'classification_report': report,
                'predictions': all_predictions,
                'ground_truth': all_ground_truth,
                'failed_samples': failed_samples
            }
        else:
            results = {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'total_samples': len(test_data),
                'successful_predictions': 0,
                'failed_predictions': len(failed_samples),
                'evaluation_time': evaluation_time,
                'avg_time_per_sample': evaluation_time / len(test_data),
                'failed_samples': failed_samples
            }
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("Batch Evaluation Results")
        print("="*60)
        
        print(f"Total Samples: {results['total_samples']}")
        print(f"Successful Predictions: {results['successful_predictions']}")
        print(f"Failed Predictions: {results['failed_predictions']}")
        print(f"Success Rate: {results['successful_predictions']/results['total_samples']*100:.2f}%")
        print(f"Evaluation Time: {results['evaluation_time']:.2f} seconds")
        print(f"Average Time per Sample: {results['avg_time_per_sample']:.3f} seconds")
        
        print("\nPerformance Metrics:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        
        if 'classification_report' in results:
            print("\nTop 15 Classes by Support:")
            report = results['classification_report']
            
            class_metrics = []
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'support' in metrics:
                    class_metrics.append((class_name, metrics))
            
            class_metrics.sort(key=lambda x: x[1]['support'], reverse=True)
            
            print(f"{'Class':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 70)
            
            for class_name, metrics in class_metrics[:15]:
                print(f"{class_name:<30} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<10}")
    
    def save_failed_predictions(self, results: Dict):
        """Save failed predictions to a separate file."""
        failed_predictions_file = os.path.join(self.output_dir, "failed_predictions.json")
        
        # Prepare failed predictions for saving
        failed_data = []
        for failed in results.get('failed_samples', []):
            failed_entry = {
                'index': failed.get('index', 'unknown'),
                'error': failed.get('error', ''),
                'predicted': failed.get('predicted', None),
                'ground_truth': failed.get('ground_truth', None),
                'sample': failed.get('sample', {})
            }
            failed_data.append(failed_entry)
        
        with open(failed_predictions_file, 'w', encoding='utf-8') as f:
            json.dump(failed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Failed predictions saved to: {failed_predictions_file}")
        print(f"Total failed predictions: {len(failed_data)}")
    
    def save_error_predictions(self, results: Dict):
        """Save error predictions (wrong predictions) to a separate file."""
        if 'predictions' not in results or 'ground_truth' not in results:
            print("No predictions available for error analysis")
            return
        
        error_predictions_file = os.path.join(self.output_dir, "error_predictions.json")
        
        # Find wrong predictions
        error_data = []
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            if pred != gt:
                error_entry = {
                    'index': i,
                    'predicted_intent': pred,
                    'predicted_label': self.business_types.get(pred, f"Unknown_{pred}"),
                    'ground_truth_intent': gt,
                    'ground_truth_label': self.business_types.get(gt, f"Unknown_{gt}"),
                    'error_type': 'wrong_prediction'
                }
                error_data.append(error_entry)
        
        with open(error_predictions_file, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        
        print(f"Error predictions saved to: {error_predictions_file}")
        print(f"Total error predictions: {len(error_data)}")
    
    def save_confusion_matrix(self, results: Dict):
        """Save confusion matrix plot."""
        if 'predictions' not in results or 'ground_truth' not in results:
            print("No predictions available for confusion matrix")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            
            # Create confusion matrix
            cm = confusion_matrix(results['ground_truth'], results['predictions'])
            
            # Get class labels
            unique_classes = sorted(set(results['ground_truth'] + results['predictions']))
            class_labels = [self.business_types.get(i, f"Class_{i}") for i in unique_classes]
            
            # Create figure
            plt.figure(figsize=(20, 16))
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_labels, yticklabels=class_labels)
            plt.title('Confusion Matrix - CMCC-34 Intent Classification', fontsize=16, pad=20)
            plt.xlabel('Predicted Intent', fontsize=14)
            plt.ylabel('True Intent', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            confusion_matrix_file = os.path.join(self.output_dir, "confusion_matrix.png")
            plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Confusion matrix saved to: {confusion_matrix_file}")
            
        except ImportError:
            print("matplotlib or seaborn not available. Skipping confusion matrix plot.")
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def save_detailed_analysis(self, results: Dict):
        """Save detailed analysis including per-class performance."""
        analysis_file = os.path.join(self.output_dir, "detailed_analysis.json")
        
        analysis = {
            'summary': {
                'total_samples': results.get('total_samples', 0),
                'successful_predictions': results.get('successful_predictions', 0),
                'failed_predictions': results.get('failed_predictions', 0),
                'accuracy': results.get('accuracy', 0.0),
                'f1_macro': results.get('f1_macro', 0.0),
                'f1_weighted': results.get('f1_weighted', 0.0)
            },
            'per_class_performance': {},
            'error_analysis': {
                'most_confused_pairs': [],
                'class_with_most_errors': None,
                'class_with_least_errors': None
            }
        }
        
        # Add per-class performance
        if 'classification_report' in results:
            for class_name, metrics in results['classification_report'].items():
                if isinstance(metrics, dict):
                    analysis['per_class_performance'][class_name] = {
                        'precision': float(metrics.get('precision', 0.0)),
                        'recall': float(metrics.get('recall', 0.0)),
                        'f1_score': float(metrics.get('f1-score', 0.0)),
                        'support': int(metrics.get('support', 0))
                    }
        
        # Find most confused pairs
        if 'predictions' in results and 'ground_truth' in results:
            from collections import Counter
            error_pairs = []
            for pred, gt in zip(results['predictions'], results['ground_truth']):
                if pred != gt:
                    error_pairs.append((gt, pred))
            
            error_pair_counts = Counter(error_pairs)
            most_confused = error_pair_counts.most_common(10)
            
            analysis['error_analysis']['most_confused_pairs'] = [
                {
                    'true_intent': self.business_types.get(pair[0], f"Class_{pair[0]}"),
                    'predicted_intent': self.business_types.get(pair[1], f"Class_{pair[1]}"),
                    'count': count
                }
                for pair, count in most_confused
            ]
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed analysis saved to: {analysis_file}")
    
    def save_final_results(self, results: Dict):
        """Save final evaluation results and additional analysis."""
        final_results_file = os.path.join(self.output_dir, "final_evaluation_results.json")
        
        # Prepare results for saving
        save_results = results.copy()
        if 'classification_report' in save_results:
            # Convert numpy types to native Python types
            for class_name, metrics in save_results['classification_report'].items():
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, np.float64):
                            metrics[key] = float(value)
                        elif isinstance(value, np.int64):
                            metrics[key] = int(value)
        
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        
        print(f"Final results saved to: {final_results_file}")
        
        # Save additional analysis
        self.save_failed_predictions(results)
        self.save_error_predictions(results)
        self.save_confusion_matrix(results)
        self.save_detailed_analysis(results)


def main():
    """Main function for batch evaluation."""
    # Configuration
    BASE_MODEL_PATH = "THUDM/GLM-4-9B-0414"
    FINETUNED_PATH = "finetune/output/cmcc34_qlora/checkpoint-5000"
    TEST_FILE = "finetune/data/cmcc-34/test.jsonl"
    OUTPUT_DIR = "evaluation_output"
    
    # Create evaluator
    evaluator = BatchEvaluator(
        base_model_path=BASE_MODEL_PATH,
        finetuned_path=FINETUNED_PATH,
        test_file=TEST_FILE,
        output_dir=OUTPUT_DIR,
        use_4bit=True
    )
    
    # Load model
    evaluator.load_model()
    
    # Load test data
    test_data = evaluator.load_test_data()
    
    # Batch evaluation
    # You can adjust batch_size based on your GPU memory
    # For 24GB GPU, batch_size=50-100 should work well
    results = evaluator.evaluate_batch(test_data, batch_size=50, start_batch=0)
    
    # Print results
    evaluator.print_results(results)
    
    # Save final results
    evaluator.save_final_results(results)
    
    print("\nBatch evaluation completed!")


if __name__ == "__main__":
    main()