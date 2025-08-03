import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from local_llm_client import LocalGLM4Client, MockLocalLLM
from config import MODEL_PATH, EMBEDDING_MODEL, EMBEDDING_MODEL_PATH, LOCAL_GLM4_URL, VECTOR_STORE_PATH, OFFLINE_MODE
from prompts import COMPLIANCE_ADVISOR_PROMPT

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG问答系统 - 整合检索和生成功能"""
    
    def __init__(self, 
                 vector_store_path: str = VECTOR_STORE_PATH,
                 model_name: str = EMBEDDING_MODEL,
                 llm_model: str = MODEL_PATH,
                 openai_api_key: Optional[str] = None,
                 use_local_glm4: bool = True):
        
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.llm_model = llm_model
        
        # 初始化嵌入模型
        if OFFLINE_MODE:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                cache_folder="~/.cache/huggingface/hub",
                model_kwargs={"local_files_only": True}
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # 初始化LLM
        if use_local_glm4:
            # 使用本地GLM4模型
            glm4_client = LocalGLM4Client()
            if glm4_client.test_connection():
                logger.info("成功连接到本地GLM4服务器")
                self.llm = MockLocalLLM(client=glm4_client)
            else:
                logger.warning("无法连接到本地GLM4服务器，使用模拟LLM")
                self.llm = MockLocalLLM()
        elif openai_api_key:
            self.llm = ChatOpenAI(
                model_name=llm_model,
                temperature=0.1,
                openai_api_key=openai_api_key
            )
        else:
            # 使用模拟LLM
            self.llm = MockLocalLLM()
        
        self.vector_store = None
        self.qa_chain = None
        
    def load_vector_store(self) -> FAISS:
        """加载向量存储"""
        try:
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("向量存储加载成功")
            return self.vector_store
        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")
            raise
    
    def create_qa_chain(self) -> RetrievalQA:
        """创建问答链"""
        if self.vector_store is None:
            self.load_vector_store()
        
        # 创建提示模板
        template = COMPLIANCE_ADVISOR_PROMPT

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 创建检索QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("问答链创建成功")
        return self.qa_chain
    
    def answer_question(self, question: str, stream: bool = False) -> Dict[str, Any]:
        """回答问题"""
        if self.qa_chain is None:
            self.create_qa_chain()
        
        try:
            if stream:
                # 流式输出 - 直接调用LLM
                print("答案: ", end="", flush=True)
                # 获取相关文档
                relevant_docs = self.vector_store.similarity_search(question, k=3)
                
                # 构建提示
                context = "\n".join([doc.page_content for doc in relevant_docs])
                prompt = COMPLIANCE_ADVISOR_PROMPT.format(context=context, question=question)
                
                # 流式调用LLM
                answer = self.llm._call(prompt, stream=True)
                print()  # 换行
            else:
                # 普通输出
                result = self.qa_chain({"query": question})
                answer = result["result"]
            
            # 获取相关文档
            relevant_docs = self.vector_store.similarity_search(question, k=3)
            
            return {
                "question": question,
                "answer": answer,
                "relevant_documents": [
                    {
                        "content": self._clean_text(doc.page_content[:300]) + "...",
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ]
            }
            
        except Exception as e:
            logger.error(f"回答问题时出错: {e}")
            return {
                "question": question,
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "relevant_documents": []
            }
    
    def _clean_text(self, text: str) -> str:
        """清理文本中的多余空格"""
        import re
        # 移除所有多余的空格，包括中文字符间的空格
        text = re.sub(r'\s+', '', text)
        # 在中文字符间添加适当的空格
        text = re.sub(r'([。，；：！？）】》」』】）])\s*([^。，；：！？）】》」』】）\s])', r'\1 \2', text)
        text = re.sub(r'([^。，；：！？（【《「『【（\s])\s*([（【《「『【（])', r'\1 \2', text)
        # 移除行首行尾空格
        text = text.strip()
        return text
    
    def _extract_document_info(self, doc) -> str:
        """提取文档信息，包括文件名和可能的章节信息"""
        source_file = doc.metadata.get('source_file', '未知文档')
        
        # 尝试从文档内容中提取章节信息
        content = doc.page_content
        import re
        
        # 查找常见的章节模式
        chapter_patterns = [
            r'第(\d+)条',  # 第X条
            r'第(\d+)章',  # 第X章
            r'第(\d+)节',  # 第X节
            r'(\d+\.\d+\.\d+)',  # X.X.X格式
            r'(\d+\.\d+)',  # X.X格式
            r'第(\d+)部分',  # 第X部分
        ]
        
        for pattern in chapter_patterns:
            match = re.search(pattern, content[:500])  # 在前500字符中查找
            if match:
                return f"{source_file} {match.group(0)}"
        
        # 如果没有找到章节信息，只返回文件名
        return source_file
    
    def batch_answer_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量回答问题"""
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
        return results

class MockLLM:
    """模拟LLM，用于测试"""
    
    def __call__(self, prompt: str) -> str:
        """模拟LLM响应"""
        if "上市公司" in prompt and "监管" in prompt:
            return "根据相关法律法规，上市公司需要遵守严格的监管要求，包括信息披露、公司治理、财务报告等方面的规定。"
        elif "信息披露" in prompt:
            return "信息披露是上市公司的重要义务，包括定期报告和临时报告，确保投资者能够及时、准确、完整地了解公司信息。"
        elif "独立董事" in prompt:
            return "独立董事是上市公司治理结构的重要组成部分，负责监督公司运作，保护中小股东利益，确保公司合规经营。"
        else:
            return "根据提供的文档内容，我可以为您提供相关信息。请具体说明您想了解的法律法规问题。"

def main():
    """主函数 - 测试RAG系统"""
    # 初始化RAG系统
    rag = RAGSystem()
    
    # 测试问题
    test_questions = [
        "上市公司监管有哪些要求？",
        "信息披露的要求是什么？",
        "独立董事的职责是什么？",
        "什么是内幕信息知情人登记管理制度？"
    ]
    
    print("RAG问答系统测试")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        result = rag.answer_question(question)
        print(f"答案: {result['answer']}")
        
        if result['relevant_documents']:
            print("相关文档:")
            for j, doc in enumerate(result['relevant_documents'], 1):
                print(f"  {j}. {doc['metadata']['source_file']}")
                print(f"     {doc['content']}")

if __name__ == "__main__":
    main() 