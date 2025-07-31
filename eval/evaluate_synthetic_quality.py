#!/usr/bin/env python3
"""
Simple synthetic data quality evaluation using DeepSeek API.
Follows KISS principle - Keep It Simple, Stupid.
"""

import json
import csv
import time
import os
from typing import List, Dict, Tuple
import requests
from tqdm import tqdm


class DialogueQualityEvaluator:
    """Simple evaluator for customer service dialogue quality using DeepSeek."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Business type mapping for reference
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
    
    def call_deepseek_api(self, messages: List[Dict]) -> str:
        """Call DeepSeek API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"API call failed after {max_retries} attempts: {e}")
                    return "ERROR"
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def evaluate_single_sample(self, dialogue_text: str, label: str) -> Dict:
        """Evaluate quality of a single customer service dialogue sample."""
        
        # Customer service dialogue quality evaluation prompt
        prompt = f"""客服对话样本质量评估框架（核心维度）

### 核心分析维度

1. 语义一致性（权重40%）
- 严格匹配标注的子类意图
- 检查意图边界清晰度（如区分"咨询"vs"办理"）
- 是否存在多意图混合（需明确主意图）

2. 上下文完整性（权重35%）
- 必须包含用户输入和客服响应
- 关键要素检查（业务名称/账号信息/时间要素等）
- 多轮对话需保持上下文连贯性

3. 语言自然性（权重25%）
- 真实口语特征（如停顿词、合理语法错误）
- 领域术语使用准确性
- 情绪表达合理性（抱怨/焦急等程度）

### 评分标准
9-10分：无需修改的理想样本
7-8分：需轻微优化的可用样本
5-6分：需要中等修改的样本
3-4分：存在严重缺陷的样本
1-2分：应丢弃的无效样本

### 输出格式要求
{{
"原始文本": "原始对话文本",
"标注标签": "当前标注标签",
"质量评分": 9-10,
"评分置信度": 0.5-1.0,
"维度分析": {{
"语义一致性": {{"得分": 9-10, "证据": "分析内容"}},
"上下文完整性": {{"得分": 9-10, "证据": "分析内容"}},
"语言自然性": {{"得分": 9-10, "证据": "分析内容"}}
}},
"改进建议": ["建议1", "建议2"]
}}

请评估以下客服对话样本：

对话文本: {dialogue_text[:500]}...
标注标签: {label}

请严格按照上述JSON格式输出评估结果。"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.call_deepseek_api(messages)
        

        
        # Parse JSON response
        scores = self.parse_json_scores(response, dialogue_text, label)
        scores["raw_response"] = response
        
        return scores
    
    def parse_json_scores(self, response: str, dialogue_text: str, label: str) -> Dict:
        """Parse JSON scores from API response."""
        scores = {
            "original_text": dialogue_text[:],
            "label": label,
            "quality_score": 0,
            "confidence": 0.0,
            "dimension_analysis": {
                "semantic_consistency": {"score": 0, "evidence": ""},
                "context_completeness": {"score": 0, "evidence": ""},
                "language_naturalness": {"score": 0, "evidence": ""}
            },
            "improvement_suggestions": [],
            "overall_rating": "无法解析"
        }
        
        try:
            # Try to extract JSON from response (handle markdown formatting)
            import re
            
            # Remove markdown code blocks if present
            response_clean = response.replace('```json', '').replace('```', '').strip()
            
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                scores["quality_score"] = result.get("质量评分", 0)
                scores["confidence"] = result.get("评分置信度", 0.0)
                
                dimension_analysis = result.get("维度分析", {})
                scores["dimension_analysis"]["semantic_consistency"] = {
                    "score": dimension_analysis.get("语义一致性", {}).get("得分", 0),
                    "evidence": dimension_analysis.get("语义一致性", {}).get("证据", "")
                }
                scores["dimension_analysis"]["context_completeness"] = {
                    "score": dimension_analysis.get("上下文完整性", {}).get("得分", 0),
                    "evidence": dimension_analysis.get("上下文完整性", {}).get("证据", "")
                }
                scores["dimension_analysis"]["language_naturalness"] = {
                    "score": dimension_analysis.get("语言自然性", {}).get("得分", 0),
                    "evidence": dimension_analysis.get("语言自然性", {}).get("证据", "")
                }
                
                scores["improvement_suggestions"] = result.get("改进建议", [])
                scores["overall_rating"] = "解析成功"
            else:
                scores["overall_rating"] = "未找到JSON格式"
        except Exception as e:
            scores["overall_rating"] = f"JSON解析失败: {str(e)}"
        
        return scores
    
    def load_data(self, json_file: str, sample_size: int = 50) -> List[Dict]:
        """Load dialogue samples from extracted JSON file for quality evaluation."""
        data = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            synthetic_samples = json.load(f)
        
        for sample in synthetic_samples:
            data.append({
                'dialogue_text': sample['dialogue_text'],
                'label': sample['label'],
                'is_synthetic': True,
                'row_number': sample.get('row_number', 0)
            })
        
        # Sample data to avoid too many API calls
        import random
        random.shuffle(data)
        return data[:sample_size]
    
    def evaluate_batch(self, data: List[Dict]) -> List[Dict]:
        """Evaluate a batch of dialogue samples."""
        results = []
        
        for i, sample in enumerate(tqdm(data, desc="Evaluating samples")):
            result = self.evaluate_single_sample(
                sample['dialogue_text'], 
                sample['label']
            )
            result['sample_id'] = i
            result['is_synthetic'] = sample['is_synthetic']
            results.append(result)
            
            # Rate limiting
            time.sleep(1)
        
        return results
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate overall statistics."""
        if not results:
            return {}
        
        # Calculate averages for new dimensions
        quality_scores = [r['quality_score'] for r in results if r['quality_score'] > 0]
        confidence_scores = [r['confidence'] for r in results if r['confidence'] > 0]
        
        semantic_scores = [r['dimension_analysis']['semantic_consistency']['score'] for r in results 
                          if r['dimension_analysis']['semantic_consistency']['score'] > 0]
        context_scores = [r['dimension_analysis']['context_completeness']['score'] for r in results 
                         if r['dimension_analysis']['context_completeness']['score'] > 0]
        language_scores = [r['dimension_analysis']['language_naturalness']['score'] for r in results 
                          if r['dimension_analysis']['language_naturalness']['score'] > 0]
        
        stats = {
            "total_samples": len(results),
            "synthetic_samples": len(results),
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "avg_semantic_consistency": sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0,
            "avg_context_completeness": sum(context_scores) / len(context_scores) if context_scores else 0,
            "avg_language_naturalness": sum(language_scores) / len(language_scores) if language_scores else 0,
            "overall_avg": 0
        }
        
        valid_scores = [s for s in [stats["avg_semantic_consistency"], stats["avg_context_completeness"], 
                                   stats["avg_language_naturalness"]] if s > 0]
        if valid_scores:
            stats["overall_avg"] = sum(valid_scores) / len(valid_scores)
        
        return stats
    
    def save_results(self, results: List[Dict], stats: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        output = {
            "statistics": stats,
            "detailed_results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, stats: Dict):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("SYNTHETIC DIALOGUE QUALITY EVALUATION SUMMARY")
        print("="*50)
        print(f"Total synthetic samples evaluated: {stats['total_samples']}")
        print(f"Average Quality Score: {stats['avg_quality_score']:.2f}/10")
        print(f"Average Confidence: {stats['avg_confidence']:.2f}")
        print(f"Average Semantic Consistency: {stats['avg_semantic_consistency']:.2f}/10")
        print(f"Average Context Completeness: {stats['avg_context_completeness']:.2f}/10")
        print(f"Average Language Naturalness: {stats['avg_language_naturalness']:.2f}/10")
        print(f"Overall Average Score: {stats['overall_avg']:.2f}/10")
        print("="*50)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate synthetic dialogue quality using DeepSeek")
    parser.add_argument("--api_key", default="sk-23e7bce10d724d6792e6adba6cf60b17", help="DeepSeek API key")
    parser.add_argument("--data_file", default="output_synthetic_quality/synthetic_data.json", help="Path to extracted JSON data file")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--output_file", default="output_synthetic_quality/results.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DialogueQualityEvaluator(args.api_key)
    
    # Load data
    print(f"Loading synthetic data from {args.data_file}...")
    data = evaluator.load_data(args.data_file, args.sample_size)
    print(f"Loaded {len(data)} synthetic samples for evaluation")
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluator.evaluate_batch(data)
    
    # Calculate statistics
    stats = evaluator.calculate_statistics(results)
    
    # Print summary
    evaluator.print_summary(stats)
    
    # Save results
    evaluator.save_results(results, stats, args.output_file)


if __name__ == "__main__":
    main() 