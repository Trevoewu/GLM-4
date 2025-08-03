#!/usr/bin/env python3
"""
RAG问答系统主程序
基于本地知识库的问答系统，通过检索外部知识库增强模型处理知识密集型任务的能力
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """设置环境"""
    # 添加当前目录到Python路径
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # 导入配置文件
    try:
        from config import setup_environment as setup_config_env
        setup_config_env()
    except ImportError:
        pass
    
    # 创建必要的目录
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def run_data_processing():
    """运行数据预处理"""
    logger.info("开始数据预处理...")
    
    from data_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    
    # 处理文档
    processed_data = processor.process_documents()
    
    # 保存处理后的数据
    processor.save_processed_data("output/processed_documents.json")
    
    # 创建问答对
    qa_pairs = processor.create_qa_pairs("output/qa_pairs.json")
    
    logger.info(f"数据预处理完成！处理了 {len(processed_data)} 个文本片段，创建了 {len(qa_pairs)} 个问答对")
    return True

def run_knowledge_base_building():
    """运行知识库构建"""
    logger.info("开始构建知识库...")
    
    from knowledge_base import KnowledgeBase
    
    kb = KnowledgeBase()
    
    # 加载处理后的文档
    documents = kb.load_processed_documents("output/processed_documents.json")
    
    # 构建向量存储
    vector_store = kb.build_vector_store(documents, "output/vector_store")
    
    # 测试搜索功能
    test_query = "上市公司信息披露要求"
    similar_docs = kb.search_similar_documents(test_query, k=3)
    
    logger.info(f"知识库构建完成！向量存储包含 {len(documents)} 个文档片段")
    logger.info(f"测试查询 '{test_query}' 找到 {len(similar_docs)} 个相关文档")
    
    return True

def run_rag_system():
    """运行RAG问答系统"""
    logger.info("启动RAG问答系统...")
    
    from rag_system import RAGSystem
    
    # 初始化RAG系统
    rag = RAGSystem(vector_store_path="output/vector_store")
    
    # 测试问答功能
    test_questions = [
        "上市公司监管有哪些要求？",
        "信息披露的要求是什么？",
        "独立董事的职责是什么？",
        "什么是内幕信息知情人登记管理制度？"
    ]
    
    print("\n" + "="*60)
    print("RAG问答系统测试")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        result = rag.answer_question(question)
        print(f"答案: {result['answer']}")
        
        if result['relevant_documents']:
            print("相关文档:")
            for j, doc in enumerate(result['relevant_documents'], 1):
                print(f"  {j}. {doc['metadata']['source_file']}")
    
    return True

def run_evaluation():
    """运行评估"""
    logger.info("开始评估RAG系统...")
    
    from evaluator import RAGEvaluator, create_test_qa_pairs
    from rag_system import RAGSystem
    
    # 初始化评估器
    evaluator = RAGEvaluator()
    
    # 创建测试问答对
    qa_pairs = create_test_qa_pairs()
    
    # 保存测试问答对
    with open("output/test_qa_pairs.json", 'w', encoding='utf-8') as f:
        import json
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    # 初始化RAG系统
    rag_system = RAGSystem(vector_store_path="output/vector_store")
    
    # 评估RAG系统
    results = evaluator.evaluate_batch(qa_pairs, rag_system)
    
    # 保存评估结果
    evaluator.save_evaluation_results(results, "output/evaluation_results.json")
    
    # 打印评估摘要
    evaluator.print_evaluation_summary(results)
    
    return True

def interactive_mode():
    """交互模式"""
    logger.info("进入交互模式...")
    
    from rag_system import RAGSystem
    
    # 初始化RAG系统
    rag = RAGSystem(vector_store_path="output/vector_store")
    
    print("\n" + "="*60)
    print("RAG问答系统 - 交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("="*60)
    
    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not question:
                continue
            
            # 回答问题（启用流式输出）
            result = rag.answer_question(question, stream=True)
            
            # 答案已经在流式输出中显示，这里只需要显示相关文档
            if not result['answer'].startswith("答案: "):
                print(f"\n答案: {result['answer']}")
            
            if result['relevant_documents']:
                print("\n相关文档:")
                for i, doc in enumerate(result['relevant_documents'], 1):
                    print(f"  {i}. {doc['metadata']['source_file']}")
                    print(f"     {doc['content']}")
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            logger.error(f"处理问题时出错: {e}")
            print(f"抱歉，处理您的问题时出现错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG问答系统")
    parser.add_argument("--mode", choices=["process", "build", "rag", "evaluate", "interactive", "full"], 
                       default="full", help="运行模式")
    parser.add_argument("--data-dir", default="dataset/法律法规", help="数据目录")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 根据模式运行相应的功能
    if args.mode == "process":
        run_data_processing()
    elif args.mode == "build":
        run_knowledge_base_building()
    elif args.mode == "rag":
        run_rag_system()
    elif args.mode == "evaluate":
        run_evaluation()
    elif args.mode == "interactive":
        interactive_mode()
    elif args.mode == "full":
        # 完整流程
        logger.info("开始完整RAG系统构建流程...")
        
        # 1. 数据预处理
        if not run_data_processing():
            logger.error("数据预处理失败")
            return
        
        # 2. 知识库构建
        if not run_knowledge_base_building():
            logger.error("知识库构建失败")
            return
        
        # 3. RAG系统测试
        if not run_rag_system():
            logger.error("RAG系统测试失败")
            return
        
        # 4. 评估
        if not run_evaluation():
            logger.error("评估失败")
            return
        
        logger.info("RAG系统构建完成！")
        
        # 5. 进入交互模式
        print("\n是否进入交互模式？(y/n): ", end="")
        if input().lower().startswith('y'):
            interactive_mode()

if __name__ == "__main__":
    main() 