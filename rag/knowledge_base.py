import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from config import EMBEDDING_MODEL, EMBEDDING_MODEL_PATH, OFFLINE_MODE
except ImportError:
    EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
    EMBEDDING_MODEL_PATH = "shibing624/text2vec-base-chinese"
    OFFLINE_MODE = True

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """知识库构建器 - 负责文档向量化和存储"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        if OFFLINE_MODE:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                cache_folder="~/.cache/huggingface/hub",
                model_kwargs={"local_files_only": True}
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.documents = []
        
    def load_processed_documents(self, file_path: str = "processed_documents.json") -> List[Dict[str, Any]]:
        """加载处理后的文档"""
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.info(f"加载了 {len(documents)} 个文档片段")
        return documents
    
    def create_texts_and_metadatas(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """创建文本和元数据"""
        texts = []
        metadatas = []
        
        for doc in documents:
            texts.append(doc['content'])
            metadatas.append({
                'source_file': doc['source_file'],
                'source_dir': doc['source_dir'],
                'segment_id': doc['segment_id'],
                'file_type': doc['file_type'],
                'length': doc['length']
            })
        
        return texts, metadatas
    
    def build_vector_store(self, documents: List[Dict[str, Any]], 
                          save_path: str = "vector_store") -> FAISS:
        """构建向量存储"""
        logger.info("开始构建向量存储...")
        
        # 创建文本和元数据
        texts, metadatas = self.create_texts_and_metadatas(documents)
        
        # 使用LangChain的FAISS向量存储
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # 保存向量存储
        self.vector_store.save_local(save_path)
        logger.info(f"向量存储已保存到 {save_path}")
        
        return self.vector_store
    
    def load_vector_store(self, load_path: str = "vector_store") -> FAISS:
        """加载向量存储"""
        self.vector_store = FAISS.load_local(
            load_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"向量存储已从 {load_path} 加载")
        return self.vector_store
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先构建或加载向量存储")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        similar_docs = []
        for doc, score in results:
            similar_docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            })
        
        return similar_docs
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """获取向量存储信息"""
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "index_type": "faiss",
            "embedding_model": self.model_name,
            "document_count": len(self.vector_store.docstore._dict)
        }

def main():
    """主函数 - 构建知识库"""
    # 初始化知识库
    kb = KnowledgeBase()
    
    # 加载处理后的文档
    documents = kb.load_processed_documents()
    
    # 构建向量存储
    vector_store = kb.build_vector_store(documents)
    
    # 测试搜索功能
    test_query = "上市公司信息披露要求"
    similar_docs = kb.search_similar_documents(test_query, k=3)
    
    print(f"\n测试查询: {test_query}")
    print(f"找到 {len(similar_docs)} 个相关文档:")
    
    for i, doc in enumerate(similar_docs, 1):
        print(f"\n{i}. 相似度: {doc['similarity_score']:.4f}")
        print(f"   来源: {doc['metadata']['source_file']}")
        print(f"   内容: {doc['content'][:200]}...")
    
    # 获取向量存储信息
    info = kb.get_vector_store_info()
    print(f"\n向量存储信息: {info}")

if __name__ == "__main__":
    main() 