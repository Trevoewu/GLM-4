#!/usr/bin/env python3
"""
合规顾问RAG系统CLI演示程序
基于现有交互模式的简洁演示界面
"""

import os
import sys
import logging
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

def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    合规顾问RAG系统演示                        ║
║                                                              ║
║  🎯 基于本地知识库的智能问答系统                              ║
║  📚 支持法律法规、监管要求等专业领域知识                      ║
║  🤖 集成大语言模型，提供准确、专业的合规建议                  ║
║  🔍 智能检索相关文档，提供可追溯的答案                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_help():
    """打印帮助信息"""
    help_text = """
📖 帮助信息:

💡 提问技巧:
   • 使用具体、明确的问题
   • 可以询问法规要求、合规流程等
   • 支持中文提问

🔍 示例问题:
   • "上市公司信息披露要求是什么？"
   • "独立董事的职责有哪些？"
   • "内幕信息管理有哪些规定？"
   • "关联交易披露标准是什么？"

⚙️ 命令:
   • help/h - 显示此帮助信息
   • quit/q/exit - 退出系统

📚 系统特点:
   • 基于法律法规知识库
   • 智能检索相关文档
   • 提供可追溯的答案
   • 支持专业合规咨询
"""
    print(help_text)

def interactive_demo():
    """交互式演示功能"""
    print("\n💬 进入交互式问答模式...")
    print("💡 提示: 输入 'help' 查看帮助，输入 'quit' 退出")
    print("=" * 60)
    
    try:
        from rag_system import RAGSystem
        
        # 初始化RAG系统
        print("📚 正在加载知识库...")
        rag = RAGSystem(vector_store_path="output/vector_store")
        print("✅ 知识库加载完成！")
        
        while True:
            try:
                question = input("\n❓ 请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 退出交互模式")
                    break
                
                if question.lower() in ['help', '帮助', 'h']:
                    print_help()
                    continue
                
                if not question:
                    continue
                
                print("🤔 正在思考...")
                
                # 回答问题（启用流式输出）
                result = rag.answer_question(question, stream=True)
                
                # 答案已经在流式输出中显示，这里只需要显示相关文档
                if not result['answer'].startswith("答案: "):
                    print(f"\n答案: {result['answer']}")
                
                if result['relevant_documents']:
                    print("\n📄 相关文档:")
                    for i, doc in enumerate(result['relevant_documents'], 1):
                        print(f"  {i}. {doc['metadata']['source_file']}")
                        print(f"     {doc['content']}")
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n👋 退出交互模式")
                break
            except Exception as e:
                print(f"❌ 处理问题时出现错误: {e}")
    
    except Exception as e:
        print(f"❌ 初始化RAG系统失败: {e}")
        print("请确保已运行完整的数据处理流程")

def main():
    """主函数"""
    # 设置环境
    setup_environment()
    
    # 打印欢迎横幅
    print_banner()
    
    # 直接进入交互模式
    interactive_demo()

if __name__ == "__main__":
    main() 