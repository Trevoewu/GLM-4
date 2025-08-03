import os
import re
import json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import jieba
import logging

# 文档处理
from pypdf import PdfReader
from docx import Document

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器 - 负责文本抽取与清洗"""
    
    def __init__(self, data_dir: str = "dataset/法律法规"):
        self.data_dir = Path(data_dir)
        self.processed_data = []
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """从PDF文件中提取文本"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"处理PDF文件 {pdf_path} 时出错: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: Path) -> str:
        """从DOCX文件中提取文本"""
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"处理DOCX文件 {docx_path} 时出错: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s\.\,\;\:\!\?\(\)\[\]\{\}]', '', text)
        
        # 移除页眉页脚等重复内容
        text = re.sub(r'第\d+页', '', text)
        text = re.sub(r'共\d+页', '', text)
        
        # 移除多余的空行
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def segment_text(self, text: str, max_length: int = 1000) -> List[str]:
        """将长文本切分成适合向量化的片段"""
        if len(text) <= max_length:
            return [text]
        
        segments = []
        sentences = re.split(r'[。！？\n]', text)
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_segment) + len(sentence) <= max_length:
                current_segment += sentence + "。"
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + "。"
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def process_documents(self) -> List[Dict[str, Any]]:
        """处理所有文档"""
        logger.info("开始处理文档...")
        
        # 遍历所有子目录
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir():
                continue
                
            logger.info(f"处理目录: {subdir.name}")
            
            for file_path in tqdm(list(subdir.glob("*")), desc=f"处理 {subdir.name}"):
                if not file_path.is_file():
                    continue
                
                # 提取文本
                text = ""
                if file_path.suffix.lower() == '.pdf':
                    text = self.extract_text_from_pdf(file_path)
                elif file_path.suffix.lower() in ['.docx', '.doc']:
                    text = self.extract_text_from_docx(file_path)
                else:
                    continue
                
                # 清洗文本
                cleaned_text = self.clean_text(text)
                if not cleaned_text:
                    continue
                
                # 切分文本
                segments = self.segment_text(cleaned_text)
                
                # 保存处理结果
                for i, segment in enumerate(segments):
                    self.processed_data.append({
                        'source_file': str(file_path),
                        'source_dir': subdir.name,
                        'segment_id': i,
                        'content': segment,
                        'length': len(segment),
                        'file_type': file_path.suffix.lower()
                    })
        
        logger.info(f"文档处理完成，共处理 {len(self.processed_data)} 个文本片段")
        return self.processed_data
    
    def save_processed_data(self, output_file: str = "processed_documents.json"):
        """保存处理后的数据"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        logger.info(f"处理后的数据已保存到 {output_file}")
    
    def create_qa_pairs(self, output_file: str = "qa_pairs.json") -> List[Dict[str, str]]:
        """创建问答对用于评估"""
        qa_pairs = []
        
        # 基于文档内容生成一些示例问答对
        for doc in self.processed_data[:100]:  # 取前100个文档片段
            content = doc['content']
            
            # 简单的问答对生成规则
            if "上市公司" in content and "监管" in content:
                qa_pairs.append({
                    "question": "上市公司监管有哪些要求？",
                    "answer": content[:500] + "...",
                    "source": doc['source_file']
                })
            
            if "信息披露" in content:
                qa_pairs.append({
                    "question": "信息披露的要求是什么？",
                    "answer": content[:500] + "...",
                    "source": doc['source_file']
                })
            
            if "独立董事" in content:
                qa_pairs.append({
                    "question": "独立董事的职责是什么？",
                    "answer": content[:500] + "...",
                    "source": doc['source_file']
                })
        
        # 保存问答对
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"创建了 {len(qa_pairs)} 个问答对，已保存到 {output_file}")
        return qa_pairs

def main():
    """主函数"""
    processor = DocumentProcessor()
    
    # 处理文档
    processed_data = processor.process_documents()
    
    # 保存处理后的数据
    processor.save_processed_data()
    
    # 创建问答对
    qa_pairs = processor.create_qa_pairs()
    
    print(f"数据处理完成！")
    print(f"- 处理了 {len(processed_data)} 个文本片段")
    print(f"- 创建了 {len(qa_pairs)} 个问答对")

if __name__ == "__main__":
    main() 