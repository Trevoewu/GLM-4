#!/usr/bin/env python3
"""
RAG系统聊天版Web演示启动脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """启动Streamlit聊天版Web演示"""
    
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    
    # 切换到脚本目录
    os.chdir(script_dir)
    
    print("🚀 启动RAG智能问答系统聊天版Web演示...")
    print(f"📁 工作目录: {script_dir}")
    print("🌐 启动后请在浏览器中访问: http://localhost:8501")
    print("💬 特色功能: ChatGPT风格的聊天界面")
    print("⏹️  按 Ctrl+C 停止服务")
    print("-" * 50)
    
    try:
        # 启动Streamlit聊天应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_chat_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ 未找到streamlit，请先安装: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main() 