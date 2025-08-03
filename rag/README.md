# RAG智能问答系统

基于本地知识库的智能问答系统，通过检索增强生成（RAG）技术提供准确、可靠的法律法规问答服务。

## 🎯 项目概述

本项目使用温氏法律文档构建知识库，通过"问题-答案"对检验RAG效果，最后通过Rouge-L指标评估模型效果。系统支持Web界面和聊天版两种交互方式。

## 🚀 快速开始

### 1. 安装依赖

```bash
cd rag
pip install -r requirements.txt
```

### 2. 启动Web演示

```bash
# 启动聊天版（推荐）
python run_chat_demo.py

# 或直接使用Streamlit
streamlit run streamlit_chat_app.py
```

### 3. 访问系统

在浏览器中打开：**http://localhost:8501**

## 📊 系统架构

```
rag/
├── data_processor.py      # 数据预处理模块 - 文本抽取与清洗
├── knowledge_base.py      # 知识库构建模块 - 文档向量化
├── rag_system.py         # RAG问答系统 - 检索增强生成
├── evaluator.py          # 评估模块 - Rouge-L指标评估
├── main.py              # 主程序 - 整合所有模块
├── streamlit_chat_app.py # 聊天版Web界面
├── run_chat_demo.py      # 聊天版启动脚本
├── local_llm_client.py   # 本地LLM客户端
├── config.py             # 配置文件
├── requirements.txt      # 依赖包
├── README.md            # 项目说明
└── dataset/             # 数据目录
    └── 法律法规/         # 法律法规文档
        ├── 证监会/       # 证监会相关文档
        └── 深交所/       # 深交所相关文档
```


## 🔧 功能模块

### 1. 数据预处理模块 (`data_processor.py`)

- **文本抽取**: 从PDF和DOCX文件中提取文本
- **文本清洗**: 移除特殊字符、页眉页脚等
- **文本切分**: 将长文本切分成适合向量化的片段
- **问答对生成**: 基于文档内容创建测试问答对

### 2. 知识库构建模块 (`knowledge_base.py`)

- **文档向量化**: 使用中文嵌入模型将文档转换为向量
- **向量存储**: 使用FAISS构建高效的向量索引
- **相似度搜索**: 基于向量相似度检索相关文档

### 3. RAG问答系统 (`rag_system.py`)

- **检索增强**: 结合文档检索和语言模型生成
- **问答链**: 使用LangChain构建问答流程
- **交互模式**: 支持用户交互式问答

### 4. 评估模块 (`evaluator.py`)

- **Rouge-L评估**: 使用Rouge-L指标评估答案质量
- **批量评估**: 支持批量问答对评估
- **结果分析**: 提供详细的评估报告


## 🛠️ 使用指南

### 运行完整流程

```bash
python main.py --mode full
```

### 分步运行

```bash
# 数据预处理
python main.py --mode process

# 知识库构建
python main.py --mode build

# RAG系统测试
python main.py --mode rag

# 评估
python main.py --mode evaluate

# 交互模式
python main.py --mode interactive
```

### 管理命令

```bash
# 启动聊天版服务
python run_chat_demo.py

# 后台运行
nohup streamlit run streamlit_chat_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true > logs/streamlit_chat.log 2>&1 &

# 停止服务
ps aux | grep streamlit
kill <进程ID>

# 查看日志
tail -f logs/streamlit_chat.log
```


## ⚙️ 配置说明

### 嵌入模型

默认使用 `shibing624/text2vec-base-chinese` 中文嵌入模型，您可以在 `knowledge_base.py` 中修改：

```python
kb = KnowledgeBase(model_name="your_model_name")
```

### 向量存储

使用FAISS作为向量存储，支持高效的相似度搜索。向量存储文件保存在 `output/vector_store/` 目录下。

### LLM配置

如需使用OpenAI API，请在初始化RAG系统时提供API密钥：

```python
rag = RAGSystem(openai_api_key="your_api_key")
```

## 📁 输出文件

- `output/processed_documents.json`: 处理后的文档数据
- `output/qa_pairs.json`: 生成的问答对
- `output/vector_store/`: 向量存储文件
- `output/evaluation_results.json`: 评估结果
- `output/test_qa_pairs.json`: 测试问答对

## 📊 评估指标

- **Rouge-1**: 单个词汇重叠度
- **Rouge-2**: 双词汇重叠度  
- **Rouge-L**: 最长公共子序列

