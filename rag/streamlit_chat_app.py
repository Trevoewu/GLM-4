import streamlit as st
import sys
import os
import logging
from pathlib import Path
import time # Added for streamlit_chat_app.py

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rag_system import RAGSystem
import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_rag_system():
    """初始化RAG系统"""
    try:
        rag_system = RAGSystem()
        # 预加载向量存储
        rag_system.load_vector_store()
        return rag_system
    except Exception as e:
        st.error(f"初始化RAG系统失败: {str(e)}")
        return None

def display_sources_with_streamlit(sources):
    """使用Streamlit内置组件显示文档来源"""
    if sources:
        st.markdown("**📚 相关文档来源:**")
        
        for i, source in enumerate(sources, 1):
            with st.expander(f"📄 相关文档 {i}: {source.get('title', '未知文档')}", expanded=False):
                st.markdown(f"**文档内容:**")
                st.text(source.get('content', ''))
                if source.get('metadata'):
                    st.markdown(f"**元数据:**")
                    st.json(source.get('metadata', {}))

def display_retrieval_process(rag_system, prompt, top_k):
    """显示检索过程"""
    st.markdown("**🔍 正在检索相关文档...**")
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 模拟检索步骤
    steps = [
        "正在分析问题...",
        "正在计算语义相似度...",
        "正在检索向量数据库...",
        "正在排序相关文档...",
        "检索完成！"
    ]
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(0.5)
    
    # 执行实际检索
    relevant_docs = rag_system.vector_store.similarity_search(prompt, k=top_k)
    
    # 显示检索结果
    st.success(f"✅ 找到 {len(relevant_docs)} 个相关文档")
    
    # 显示检索到的文档预览
    with st.expander("📋 检索到的文档预览", expanded=False):
        for i, doc in enumerate(relevant_docs, 1):
            st.markdown(f"**文档 {i}:**")
            st.text(doc.page_content[:200] + "...")
            if doc.metadata:
                st.markdown(f"*来源: {doc.metadata.get('source_file', '未知')}*")
            st.divider()
    
    return relevant_docs

def main():
    # 主标题
    st.title("🤖 RAG智能问答系统")
    st.markdown("基于检索增强生成的智能问答系统，为您提供准确、可靠的答案")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")
        
        rag_system = initialize_rag_system()
        
        if rag_system:
            # 参数设置
            st.subheader("🔧 参数设置")
            top_k = st.slider("检索文档数量", min_value=1, max_value=10, value=3, help="从知识库中检索的相关文档数量")
            temperature = st.slider("生成温度", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="控制答案生成的创造性")
            
            # 流式输出设置
            st.subheader("🔄 输出设置")
            use_streaming = st.checkbox("启用流式输出", value=True, help="实时显示生成过程")
            
            # 检索过程显示设置
            st.subheader("🔍 检索设置")
            show_retrieval = st.checkbox("显示检索过程", value=True, help="显示文档检索的详细过程")
            
        else:
            st.error("❌ RAG系统初始化失败")
            st.stop()
    
    # 主聊天界面
    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                display_sources_with_streamlit(message["sources"])
    
    # 聊天输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 显示助手消息
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考中..."):
                try:
                    # 显示检索过程
                    if show_retrieval:
                        relevant_docs = display_retrieval_process(rag_system, prompt, top_k)
                    else:
                        # 静默检索
                        relevant_docs = rag_system.vector_store.similarity_search(prompt, k=top_k)
                    
                    # 获取答案
                    if use_streaming:
                        # 流式输出
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        # 构建提示
                        context = "\n".join([doc.page_content for doc in relevant_docs])
                        
                        # 转换文档格式（与普通输出保持一致）
                        relevant_docs_formatted = [
                            {
                                "content": rag_system._clean_text(doc.page_content[:300]) + "...",
                                "metadata": doc.metadata
                            }
                            for doc in relevant_docs
                        ]
                        prompt_text = f"""基于以下上下文信息，请回答用户的问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

上下文信息：
{context}

问题：{prompt}

请提供准确、详细的回答："""
                        
                        # 模拟流式输出
                        try:
                            # 这里应该调用真正的流式LLM，现在先模拟
                            answer = rag_system.llm._call(prompt_text)
                            
                            # 模拟逐字显示（保持Markdown格式）
                            # 将答案按句子分割，保持Markdown格式
                            sentences = answer.split('。')
                            full_response = ""
                            
                            for i, sentence in enumerate(sentences):
                                if sentence.strip():
                                    full_response += sentence + "。"
                                    message_placeholder.markdown(full_response + "▌")
                                    time.sleep(0.2)
                            
                            # 最终显示完整的Markdown格式答案
                            message_placeholder.markdown(answer)
                            
                        except Exception as e:
                            # 如果流式输出失败，回退到普通输出
                            response = rag_system.answer_question(prompt)
                            answer = response.get("answer", "抱歉，我无法找到相关答案。")
                            relevant_docs = response.get("relevant_documents", [])
                            st.markdown(answer)
                    else:
                        # 普通输出
                        response = rag_system.answer_question(prompt)
                        answer = response.get("answer", "抱歉，我无法找到相关答案。")
                        relevant_docs = response.get("relevant_documents", [])
                        st.markdown(answer)
                    
                    # 转换文档格式
                    sources = []
                    if use_streaming:
                        # 流式输出使用已格式化的文档
                        for i, doc in enumerate(relevant_docs_formatted, 1):
                            sources.append({
                                "title": f"相关文档 {i}",
                                "content": doc.get("content", ""),
                                "metadata": doc.get("metadata", {})
                            })
                    else:
                        # 普通输出使用response中的文档
                        for i, doc in enumerate(relevant_docs, 1):
                            sources.append({
                                "title": f"相关文档 {i}",
                                "content": doc.get("content", ""),
                                "metadata": doc.get("metadata", {})
                            })
                    
                    # 显示来源
                    display_sources_with_streamlit(sources)
                    
                    # 添加助手消息到历史
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"处理问题时发生错误: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"查询错误: {str(e)}")
    
    # 底部控制按钮
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 