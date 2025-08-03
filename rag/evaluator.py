import json
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """RAG系统评估器 - 使用Rouge-L指标评估效果"""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_qa_pairs(self, file_path: str = "qa_pairs.json") -> List[Dict[str, str]]:
        """加载问答对"""
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        logger.info(f"加载了 {len(qa_pairs)} 个问答对")
        return qa_pairs
    
    def calculate_rouge_scores(self, reference: str, prediction: str) -> Dict[str, float]:
        """计算Rouge分数"""
        scores = self.scorer.score(reference, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def evaluate_single_qa(self, question: str, reference_answer: str, 
                          rag_system, **kwargs) -> Dict[str, Any]:
        """评估单个问答对"""
        try:
            # 使用RAG系统回答问题
            result = rag_system.answer_question(question)
            predicted_answer = result['answer']
            
            # 计算Rouge分数
            rouge_scores = self.calculate_rouge_scores(reference_answer, predicted_answer)
            
            return {
                'question': question,
                'reference_answer': reference_answer,
                'predicted_answer': predicted_answer,
                'rouge_scores': rouge_scores,
                'relevant_documents': result.get('relevant_documents', [])
            }
            
        except Exception as e:
            logger.error(f"评估问答对时出错: {e}")
            return {
                'question': question,
                'reference_answer': reference_answer,
                'predicted_answer': f"错误: {str(e)}",
                'rouge_scores': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'relevant_documents': []
            }
    
    def evaluate_batch(self, qa_pairs: List[Dict[str, str]], 
                      rag_system, **kwargs) -> Dict[str, Any]:
        """批量评估问答对"""
        results = []
        
        for qa_pair in tqdm(qa_pairs, desc="评估问答对"):
            result = self.evaluate_single_qa(
                question=qa_pair['question'],
                reference_answer=qa_pair['answer'],
                rag_system=rag_system
            )
            results.append(result)
        
        # 计算平均分数
        avg_scores = self.calculate_average_scores(results)
        
        return {
            'individual_results': results,
            'average_scores': avg_scores,
            'total_qa_pairs': len(qa_pairs)
        }
    
    def calculate_average_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算平均分数"""
        total_rouge1 = 0.0
        total_rouge2 = 0.0
        total_rougeL = 0.0
        valid_count = 0
        
        for result in results:
            if 'rouge_scores' in result:
                total_rouge1 += result['rouge_scores']['rouge1']
                total_rouge2 += result['rouge_scores']['rouge2']
                total_rougeL += result['rouge_scores']['rougeL']
                valid_count += 1
        
        if valid_count == 0:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        return {
            'rouge1': total_rouge1 / valid_count,
            'rouge2': total_rouge2 / valid_count,
            'rougeL': total_rougeL / valid_count
        }
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                              output_file: str = "evaluation_results.json"):
        """保存评估结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到 {output_file}")
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        avg_scores = results['average_scores']
        total_pairs = results['total_qa_pairs']
        
        print("\n" + "="*50)
        print("RAG系统评估结果")
        print("="*50)
        print(f"评估问答对数量: {total_pairs}")
        print(f"Rouge-1 平均分数: {avg_scores['rouge1']:.4f}")
        print(f"Rouge-2 平均分数: {avg_scores['rouge2']:.4f}")
        print(f"Rouge-L 平均分数: {avg_scores['rougeL']:.4f}")
        print("="*50)
        
        # 打印前几个详细结果
        print("\n前3个问答对详细结果:")
        for i, result in enumerate(results['individual_results'][:3], 1):
            print(f"\n{i}. 问题: {result['question']}")
            print(f"   参考答案: {result['reference_answer'][:100]}...")
            print(f"   预测答案: {result['predicted_answer'][:100]}...")
            print(f"   Rouge-L分数: {result['rouge_scores']['rougeL']:.4f}")

def create_test_qa_pairs() -> List[Dict[str, str]]:
    """创建测试问答对"""
    qa_pairs = [
        {
            "question": "上市公司信息披露的基本要求是什么？",
            "answer": "上市公司信息披露的基本要求包括及时性、准确性、完整性和公平性。公司应当真实、准确、完整、及时地披露信息，不得有虚假记载、误导性陈述或者重大遗漏。"
        },
        {
            "question": "独立董事的主要职责有哪些？",
            "answer": "独立董事的主要职责包括监督公司运作、保护中小股东利益、确保公司合规经营、参与重大决策等。独立董事应当独立履行职责，不受公司主要股东、实际控制人影响。"
        },
        {
            "question": "什么是内幕信息知情人登记管理制度？",
            "answer": "内幕信息知情人登记管理制度是指上市公司应当对内幕信息知情人进行登记管理，记录内幕信息知情人的身份信息、知悉内幕信息的时间、方式等，并按照规定向监管部门报告。"
        },
        {
            "question": "上市公司募集资金使用有哪些监管要求？",
            "answer": "上市公司募集资金应当严格按照招股说明书或募集说明书披露的用途使用，不得擅自改变用途。公司应当建立募集资金专项存储制度，确保募集资金的安全和有效使用。"
        },
        {
            "question": "上市公司现金分红政策有哪些规定？",
            "answer": "上市公司应当制定现金分红政策，明确分红条件、分红比例、分红时间等。公司应当在公司章程中明确现金分红政策，并在年度报告中披露分红政策的执行情况。"
        }
    ]
    return qa_pairs

def main():
    """主函数 - 评估RAG系统"""
    from rag_system import RAGSystem
    
    # 初始化评估器
    evaluator = RAGEvaluator()
    
    # 创建测试问答对
    qa_pairs = create_test_qa_pairs()
    
    # 保存测试问答对
    with open("test_qa_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    # 初始化RAG系统
    rag_system = RAGSystem()
    
    # 评估RAG系统
    results = evaluator.evaluate_batch(qa_pairs, rag_system)
    
    # 保存评估结果
    evaluator.save_evaluation_results(results)
    
    # 打印评估摘要
    evaluator.print_evaluation_summary(results)

if __name__ == "__main__":
    main() 