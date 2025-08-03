#!/usr/bin/env python3
"""
RAG系统配置文件
"""

import os
from pathlib import Path

# 模型配置
MODEL_PATH = "THUDM/GLM-4-9B-0414"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# 本地服务器配置
LOCAL_GLM4_URL = "http://localhost:8001"
LOCAL_GLM4_PORT = 8001

# 向量存储配置
VECTOR_STORE_PATH = "output/vector_store"

# 数据目录
DATA_DIR = "dataset/法律法规"

# 输出目录
OUTPUT_DIR = "output"

# 缓存目录
CACHE_DIR = Path.home() / ".cache" / "huggingface"
HUB_CACHE_DIR = CACHE_DIR / "hub"

# 离线模式配置
OFFLINE_MODE = True

# 环境变量设置
def setup_environment():
    """设置环境变量"""
    if OFFLINE_MODE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(CACHE_DIR)
        # 使用新的环境变量，避免弃用警告
        os.environ["HF_HUB_LOCAL_FILES_ONLY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR)

# 确保缓存目录存在
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 本地模型路径
EMBEDDING_MODEL_PATH = str(HUB_CACHE_DIR / "models--shibing624--text2vec-base-chinese" / "snapshots" / "183bb99aa7af74355fb58d16edf8c13ae7c5433e")

# 初始化环境
setup_environment()

# 模型检查函数
def check_models():
    """检查模型是否可用"""
    models = [
        EMBEDDING_MODEL,
        MODEL_PATH
    ]
    
    print("检查模型...")
    for model in models:
        model_dir = HUB_CACHE_DIR / f"models--{model.replace('/', '--')}"
        if model_dir.exists():
            print(f"✓ {model} - 已缓存")
        else:
            print(f"✗ {model} - 未找到")
    
    return True

if __name__ == "__main__":
    print("RAG系统配置")
    print("=" * 30)
    print(f"模型路径: {MODEL_PATH}")
    print(f"嵌入模型: {EMBEDDING_MODEL}")
    print(f"本地服务器URL: {LOCAL_GLM4_URL}")
    print(f"向量存储路径: {VECTOR_STORE_PATH}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"缓存目录: {CACHE_DIR}")
    print(f"离线模式: {OFFLINE_MODE}")
    print("=" * 30)
    
    check_models() 