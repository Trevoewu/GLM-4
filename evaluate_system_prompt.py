#!/usr/bin/env python3
"""
Evaluation script for CMCC-34 intent classification with system prompt optimization.
This script evaluates the fine-tuned model using the new system prompt approach.
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


class SystemPromptEvaluator:
    """Evaluator for system prompt optimized model."""
    
    def __init__(self, base_model_path: str, finetuned_path: str, test_file: str, 
                 output_dir: str = "evaluation_output_system_prompt", use_4bit: bool = True):
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
        
        # System prompt - same as used in training
        self.system_prompt = """你是客服意图识别专家。分析对话内容，判断用户最终意图。

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
    
    def generate_response(self, user_content: str, max_retries: int = 2) -> str:
        """Generate response for a user message with system prompt and retry logic."""
        # Try with original content first
        current_content = user_content
        original_length = len(user_content)
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": current_content}
                ]
                
                model_inputs = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(self.model.device)
                
                # Check input token length
                input_tokens = model_inputs["input_ids"].shape[1]
                if input_tokens > 2048:  # GLM-4 context limit
                    print(f"⚠️  Input too long ({input_tokens} tokens), truncating... (attempt {attempt + 1}/{max_retries + 1})")
                    # Truncate the user content to fit within context
                    max_user_tokens = 2048 - len(self.tokenizer.encode(self.system_prompt))
                    truncated_content = self.tokenizer.decode(
                        self.tokenizer.encode(current_content)[:max_user_tokens], 
                        skip_special_tokens=True
                    )
                    current_content = truncated_content
                    continue
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=1024,  # increase for intent classification
                        do_sample=True,
                        top_p=0.8,
                        temperature=0.6,
                        repetition_penalty=1.2,
                        eos_token_id=self.model.config.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                generated_ids = outputs[:, model_inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                if attempt > 0:
                    print(f"✅ Retry successful on attempt {attempt + 1}")
                
                return response.strip()
                
            except Exception as e:
                last_error = str(e)
                error_msg = str(e).lower()
                
                if "out of memory" in error_msg or "cuda" in error_msg:
                    print(f"🔄 Memory error on attempt {attempt + 1}, trying with truncated content...")
                    # Truncate content for next attempt
                    if attempt < max_retries:
                        old_length = len(current_content)
                        # Truncate to 80% of current length
                        current_content = current_content[:int(len(current_content) * 0.8)]
                        new_length = len(current_content)
                        print(f"   Truncated from {old_length} to {new_length} characters")
                    else:
                        print(f"❌ All retry attempts exhausted")
                        raise e
                else:
                    if attempt > 0:
                        print(f"❌ Retry failed on attempt {attempt + 1}: {str(e)}")
                    raise e
        
        # If all attempts failed
        print(f"💥 All {max_retries + 1} attempts failed. Original length: {original_length} chars")
        raise Exception(f"All {max_retries + 1} attempts failed. Last error: {last_error}")
    
    def evaluate_batch(self, test_data: List[Dict], batch_size: int = 50) -> Dict:
        """Evaluate data in batches."""
        print(f"Starting batch evaluation with batch_size={batch_size}")
        print(f"Total samples: {len(test_data)}")
        
        total_batches = (len(test_data) + batch_size - 1) // batch_size
        all_predictions = []
        all_ground_truth = []
        failed_samples = []
        
        start_time = time.time()
        
        for batch_num in range(total_batches):
            print(f"\nProcessing batch {batch_num + 1}/{total_batches}")
            
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
                pred_response = None
                pred_intent = None
                error_msg = None
                
                try:
                    pred_response = self.generate_response(user_message)
                    pred_intent = self.extract_intent(pred_response)
                except Exception as e:
                    print(f"Error on sample {start_idx + i}: {e}")
                    error_msg = str(e)
                
                # Check if prediction was successful
                if pred_intent is not None and gt_intent is not None:
                    batch_predictions.append(pred_intent)
                    batch_ground_truth.append(gt_intent)
                else:
                    # Add to failed samples with full information
                    failed_sample = {
                        'index': start_idx + i,
                        'predicted_response': pred_response,
                        'predicted_intent': pred_intent,
                        'ground_truth': gt_intent,
                        'sample': sample
                    }
                    
                    # Add error message if there was an exception
                    if error_msg:
                        failed_sample['error'] = error_msg
                    
                    batch_failed.append(failed_sample)
            
            # Accumulate results
            all_predictions.extend(batch_predictions)
            all_ground_truth.extend(batch_ground_truth)
            failed_samples.extend(batch_failed)
            
            # Print batch summary
            batch_accuracy = accuracy_score(batch_ground_truth, batch_predictions) if batch_predictions else 0
            print(f"Batch {batch_num + 1} - Accuracy: {batch_accuracy:.4f}, "
                  f"Successful: {len(batch_predictions)}, Failed: {len(batch_failed)}")
        
        evaluation_time = time.time() - start_time
        
        # Analyze token distribution
        token_lengths = []
        failed_token_lengths = []
        
        for sample in test_data:
            if len(sample["messages"]) == 3:
                user_message = sample["messages"][1]["content"]
            else:
                user_message = sample["messages"][0]["content"]
            
            # Count tokens in user message
            tokens = self.tokenizer.encode(user_message)
            token_lengths.append(len(tokens))
        
        # Analyze failed samples token lengths
        for failed in failed_samples:
            if 'sample' in failed:
                sample = failed['sample']
                if len(sample["messages"]) == 3:
                    user_message = sample["messages"][1]["content"]
                else:
                    user_message = sample["messages"][0]["content"]
                
                tokens = self.tokenizer.encode(user_message)
                failed_token_lengths.append(len(tokens))
        
        # Store token lengths for failed samples in the failed_samples data
        for i, failed in enumerate(failed_samples):
            if i < len(failed_token_lengths):
                failed['token_length'] = failed_token_lengths[i]
        
        # Calculate final metrics
        if len(all_predictions) > 0:
            accuracy = accuracy_score(all_ground_truth, all_predictions)
            f1_macro = f1_score(all_ground_truth, all_predictions, average='macro')
            f1_weighted = f1_score(all_ground_truth, all_predictions, average='weighted')
            
            # Classification report
            class_names = [self.business_types.get(i, f"Class_{i}") for i in sorted(set(all_ground_truth + all_predictions))]
            report = classification_report(all_ground_truth, all_predictions, target_names=class_names, output_dict=True, zero_division=0)
            
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
                'failed_samples': failed_samples,
                'test_data': test_data,  # Store test data for token analysis
                'token_analysis': {
                    'all_samples': {
                        'mean': sum(token_lengths) / len(token_lengths),
                        'median': sorted(token_lengths)[len(token_lengths)//2],
                        'max': max(token_lengths),
                        'min': min(token_lengths),
                        'std': (sum((x - sum(token_lengths)/len(token_lengths))**2 for x in token_lengths) / len(token_lengths))**0.5
                    },
                    'failed_samples': {
                        'mean': sum(failed_token_lengths) / len(failed_token_lengths) if failed_token_lengths else 0,
                        'median': sorted(failed_token_lengths)[len(failed_token_lengths)//2] if failed_token_lengths else 0,
                        'max': max(failed_token_lengths) if failed_token_lengths else 0,
                        'min': min(failed_token_lengths) if failed_token_lengths else 0,
                        'std': (sum((x - sum(failed_token_lengths)/len(failed_token_lengths))**2 for x in failed_token_lengths) / len(failed_token_lengths))**0.5 if failed_token_lengths else 0
                    }
                }
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
                'failed_samples': failed_samples,
                'test_data': test_data,  # Store test data for token analysis
                'token_analysis': {
                    'all_samples': {
                        'mean': sum(token_lengths) / len(token_lengths),
                        'median': sorted(token_lengths)[len(token_lengths)//2],
                        'max': max(token_lengths),
                        'min': min(token_lengths),
                        'std': (sum((x - sum(token_lengths)/len(token_lengths))**2 for x in token_lengths) / len(token_lengths))**0.5
                    },
                    'failed_samples': {
                        'mean': sum(failed_token_lengths) / len(failed_token_lengths) if failed_token_lengths else 0,
                        'median': sorted(failed_token_lengths)[len(failed_token_lengths)//2] if failed_token_lengths else 0,
                        'max': max(failed_token_lengths) if failed_token_lengths else 0,
                        'min': min(failed_token_lengths) if failed_token_lengths else 0,
                        'std': (sum((x - sum(failed_token_lengths)/len(failed_token_lengths))**2 for x in failed_token_lengths) / len(failed_token_lengths))**0.5 if failed_token_lengths else 0
                    }
                }
            }
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("System Prompt Evaluation Results")
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
        
        # Print token analysis if available
        if 'token_analysis' in results:
            print("\nToken Analysis:")
            token_analysis = results['token_analysis']
            all_stats = token_analysis['all_samples']
            failed_stats = token_analysis['failed_samples']
            
            print(f"All Samples - Mean: {all_stats['mean']:.1f}, Median: {all_stats['median']:.1f}, Max: {all_stats['max']:.1f}")
            if failed_stats['mean'] > 0:
                print(f"Failed Samples - Mean: {failed_stats['mean']:.1f}, Median: {failed_stats['median']:.1f}, Max: {failed_stats['max']:.1f}")
                if failed_stats['mean'] > all_stats['mean'] * 1.2:
                    print("⚠️  Failed samples are significantly longer than average - token length may be causing failures")
                else:
                    print("✅ Failed samples are not significantly longer than average")
            else:
                print("✅ No failed samples to analyze")
        
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
            # Get predicted response, handle empty strings
            pred_response = failed.get('predicted_response', None)
            if pred_response == "":
                pred_response = None
            
            failed_entry = {
                'index': failed.get('index', 'unknown'),
                'error': failed.get('error', ''),
                'predicted_response': pred_response,
                'predicted_intent': failed.get('predicted_intent', None),
                'ground_truth': failed.get('ground_truth', None),
                'token_length': failed.get('token_length', None),
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
            
            # Configure matplotlib for Chinese characters with better font fallback
            import matplotlib.font_manager as fm
            
            # Try to find a font that supports Chinese characters
            chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'SimHei', 'Microsoft YaHei', 'Source Han Sans CN']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            # Find the first available Chinese font
            selected_font = None
            for font in chinese_fonts:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
                print(f"Using Chinese font: {selected_font}")
            else:
                print("Warning: No Chinese font found. Using default font with potential rendering issues.")
                # Use a font that at least supports basic Unicode
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            
            plt.rcParams['axes.unicode_minus'] = False
            
            # Create confusion matrix
            cm = confusion_matrix(results['ground_truth'], results['predictions'])
            
            # Get class labels
            unique_classes = sorted(set(results['ground_truth'] + results['predictions']))
            
            # Use English labels if no Chinese font is available
            if not selected_font:
                class_labels = [f"Class_{i}" for i in unique_classes]
                print("Using English class labels due to font limitations")
            else:
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
    
    def save_token_analysis(self, results: Dict):
        """Save token distribution analysis and plots."""
        if 'token_analysis' not in results:
            print("No token analysis available")
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Configure matplotlib for Chinese characters
            import matplotlib.font_manager as fm
            chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'SimHei', 'Microsoft YaHei', 'Source Han Sans CN']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            selected_font = None
            for font in chinese_fonts:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Get token data
            token_analysis = results['token_analysis']
            all_stats = token_analysis['all_samples']
            failed_stats = token_analysis['failed_samples']
            
            # Get actual token length data for plotting
            all_token_lengths = []
            failed_token_lengths = []
            
            # Extract token lengths from test data and failed samples
            for sample in results.get('test_data', []):
                if len(sample["messages"]) == 3:
                    user_message = sample["messages"][1]["content"]
                else:
                    user_message = sample["messages"][0]["content"]
                tokens = self.tokenizer.encode(user_message)
                all_token_lengths.append(len(tokens))
            
            for failed in results.get('failed_samples', []):
                if 'token_length' in failed:
                    failed_token_lengths.append(failed['token_length'])
            
            # Plot 1: Token length distribution (all samples)
            if all_token_lengths:
                ax1.hist(all_token_lengths, bins=min(20, len(all_token_lengths)//5), alpha=0.7, color='blue', label='All Samples')
                ax1.axvline(all_stats['mean'], color='red', linestyle='--', label=f'Mean: {all_stats["mean"]:.1f}')
                ax1.axvline(all_stats['median'], color='green', linestyle='--', label=f'Median: {all_stats["median"]:.1f}')
                ax1.set_title('Token Length Distribution (All Samples)')
                ax1.set_xlabel('Token Count')
                ax1.set_ylabel('Frequency')
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, 'No token data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Token Length Distribution (All Samples)')
            
            # Plot 2: Token length distribution (failed samples)
            if failed_token_lengths:
                ax2.hist(failed_token_lengths, bins=min(20, max(1, len(failed_token_lengths)//3)), alpha=0.7, color='red', label='Failed Samples')
                ax2.axvline(failed_stats['mean'], color='red', linestyle='--', label=f'Mean: {failed_stats["mean"]:.1f}')
                ax2.axvline(failed_stats['median'], color='green', linestyle='--', label=f'Median: {failed_stats["median"]:.1f}')
                ax2.set_title('Token Length Distribution (Failed Samples)')
                ax2.set_xlabel('Token Count')
                ax2.set_ylabel('Frequency')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No failed samples', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Token Length Distribution (Failed Samples)')
            
            # Plot 3: Comparison box plot
            if all_token_lengths and failed_token_lengths:
                data = [all_token_lengths, failed_token_lengths]
                labels = ['All Samples', 'Failed Samples']
                ax3.boxplot(data, labels=labels)
                ax3.set_title('Token Length Comparison')
                ax3.set_ylabel('Token Count')
            else:
                ax3.text(0.5, 0.5, 'No data for comparison', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Token Length Comparison')
            
            # Plot 4: Statistics summary
            stats_text = f"""Token Analysis Summary

All Samples:
- Mean: {all_stats['mean']:.1f}
- Median: {all_stats['median']:.1f}
- Max: {all_stats['max']:.1f}
- Min: {all_stats['min']:.1f}
- Std: {all_stats['std']:.1f}

Failed Samples:
- Mean: {failed_stats['mean']:.1f}
- Median: {failed_stats['median']:.1f}
- Max: {failed_stats['max']:.1f}
- Min: {failed_stats['min']:.1f}
- Std: {failed_stats['std']:.1f}

Analysis:
- Failed samples are {'longer' if failed_stats['mean'] > all_stats['mean'] else 'shorter'} than average
- {'Token length may be causing failures' if failed_stats['mean'] > all_stats['mean'] * 1.2 else 'Token length unlikely to be the main cause'}"""
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.set_title('Token Analysis Summary')
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            token_analysis_file = os.path.join(self.output_dir, "token_analysis.png")
            plt.savefig(token_analysis_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Token analysis plot saved to: {token_analysis_file}")
            
        except ImportError:
            print("matplotlib not available. Skipping token analysis plot.")
        except Exception as e:
            print(f"Error creating token analysis plot: {e}")
    
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
        
        # Find most confused pairs and class error statistics
        if 'predictions' in results and 'ground_truth' in results:
            from collections import Counter
            error_pairs = []
            class_errors = Counter()
            
            for pred, gt in zip(results['predictions'], results['ground_truth']):
                if pred != gt:
                    error_pairs.append((gt, pred))
                    class_errors[gt] += 1  # Count errors per true class
            
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
            
            # Find classes with most and least errors
            if class_errors:
                most_error_class = class_errors.most_common(1)[0][0]
                least_error_class = min(class_errors.items(), key=lambda x: x[1])[0]
                
                analysis['error_analysis']['class_with_most_errors'] = {
                    'intent_id': most_error_class,
                    'intent_name': self.business_types.get(most_error_class, f"Class_{most_error_class}"),
                    'error_count': class_errors[most_error_class]
                }
                
                analysis['error_analysis']['class_with_least_errors'] = {
                    'intent_id': least_error_class,
                    'intent_name': self.business_types.get(least_error_class, f"Class_{least_error_class}"),
                    'error_count': class_errors[least_error_class]
                }
            
            # Add overall error statistics
            analysis['error_analysis']['total_errors'] = len(error_pairs)
            analysis['error_analysis']['error_rate'] = len(error_pairs) / len(results['predictions']) if results['predictions'] else 0
            analysis['error_analysis']['classes_with_errors'] = len(class_errors)
            analysis['error_analysis']['error_distribution'] = {
                self.business_types.get(intent_id, f"Class_{intent_id}"): count
                for intent_id, count in class_errors.most_common()
            }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed analysis saved to: {analysis_file}")
    
    def save_final_results(self, results: Dict):
        """Save final evaluation results and additional analysis."""
        final_results_file = os.path.join(self.output_dir, "system_prompt_evaluation_results.json")
        
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
        self.save_token_analysis(results)
        self.save_detailed_analysis(results)


def main():
    """Main function for system prompt evaluation."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GLM-4 CMCC-34 System Prompt Evaluation Script')
    parser.add_argument('--quick', '-q', action='store_true', 
                       help='Quick evaluation with limited samples (default: 100)')
    parser.add_argument('--samples', '-s', type=int, default=100,
                       help='Number of samples for quick evaluation (default: 100)')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                       help='Batch size for evaluation (default: 50)')
    parser.add_argument('--output-dir', '-o', type=str, default="evaluation_output_system_prompt",
                       help='Output directory for results (default: evaluation_output_system_prompt)')
    parser.add_argument('--model-path', '-m', type=str, 
                       default="finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000",
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--test-file', '-t', type=str,
                       default="finetune/data/cmcc-34/test.jsonl",
                       help='Path to test file')
    
    args = parser.parse_args()
    
    # Configuration
    BASE_MODEL_PATH = "THUDM/GLM-4-9B-0414"
    FINETUNED_PATH = args.model_path
    TEST_FILE = args.test_file
    OUTPUT_DIR = args.output_dir
    
    # Add quick suffix to output dir if quick evaluation
    if args.quick:
        OUTPUT_DIR = f"{OUTPUT_DIR}_quick"
    
    print(f"System Prompt Evaluation Configuration:")
    print(f"  Model: {FINETUNED_PATH}")
    print(f"  Test File: {TEST_FILE}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Quick Mode: {args.quick}")
    if args.quick:
        print(f"  Sample Limit: {args.samples}")
    print(f"  Batch Size: {args.batch_size}")
    print()
    
    # Create evaluator
    evaluator = SystemPromptEvaluator(
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
    
    # Limit samples for quick evaluation
    if args.quick:
        original_size = len(test_data)
        test_data = test_data[:args.samples]
        print(f"Quick evaluation: Using {len(test_data)} samples (from {original_size} total)")
    
    # Batch evaluation
    results = evaluator.evaluate_batch(test_data, batch_size=args.batch_size)
    
    # Print results
    evaluator.print_results(results)
    
    # Save final results
    evaluator.save_final_results(results)
    
    print(f"\n{'Quick ' if args.quick else ''}System prompt evaluation completed!")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()