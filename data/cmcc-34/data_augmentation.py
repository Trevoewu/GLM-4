 #!/usr/bin/env python3
"""
LLM-based data augmentation for CMCC-34 dataset to handle class imbalance.
This script uses GLM-4 or other LLMs to generate synthetic samples for underrepresented classes.
"""

import pandas as pd
import json
import os
import random
from collections import Counter
from typing import List, Dict, Tuple
import requests
import time

class CMCCDataAugmentation:
    def __init__(self):
        """初始化数据增强器"""
        self.business_types = {
            0: "咨询（含查询）业务规定",
            1: "办理取消",
            2: "咨询（含查询）业务资费", 
            3: "咨询（含查询）营销活动信息",
            4: "咨询（含查询）办理方式",
            5: "投诉（含抱怨）业务使用问题",
            6: "咨询（含查询）账户信息",
            7: "办理开通",
            8: "咨询（含查询）业务订购信息查询",
            9: "投诉（含抱怨）不知情定制问题",
            10: "咨询（含查询）产品/业务功能",
            11: "咨询（含查询）用户资料",
            12: "投诉（含抱怨）费用问题",
            13: "投诉（含抱怨）业务办理问题",
            14: "投诉（含抱怨）服务问题",
            15: "办理变更",
            16: "咨询（含查询）服务渠道信息",
            17: "投诉（含抱怨）业务规定不满",
            18: "投诉（含抱怨）营销问题",
            19: "投诉（含抱怨）网络问题",
            20: "办理停复机",
            21: "投诉（含抱怨）信息安全问题",
            22: "办理重置/修改/补发",
            23: "咨询（含查询）使用方式",
            24: "咨询（含查询）号码状态",
            25: "咨询（含查询）工单处理结果",
            26: "办理打印/邮寄",
            27: "咨询（含查询）宽带覆盖范围",
            28: "办理移机/装机/拆机",
            29: "办理缴费",
            30: "办理下载/设置",
            31: "办理补换卡",
            32: "办理销户/重开",
            33: "咨询（含查询）电商货品信息"
        }
        
        # 类别关键词映射，用于生成更准确的样本
        self.category_keywords = {
            0: ["业务规定", "规则", "政策", "条款", "规范", "标准"],
            1: ["取消", "退订", "关闭", "停用", "撤销"],
            2: ["资费", "价格", "费用", "收费", "tariff", "计费"],
            3: ["营销活动", "优惠", "促销", "活动", "特价", "折扣"],
            4: ["办理方式", "如何办理", "怎么办", "办理流程", "操作步骤"],
            5: ["使用问题", "故障", "无法使用", "不能用", "异常"],
            6: ["账户信息", "余额", "账单", "消费", "话费"],
            7: ["开通", "激活", "启用", "办理", "申请"],
            8: ["订购信息", "已订购", "订购查询", "业务查询"],
            9: ["不知情定制", "莫名其妙", "没有订购", "自动订购"],
            10: ["产品功能", "业务功能", "如何使用", "功能介绍"],
            11: ["用户资料", "个人信息", "身份信息", "资料修改"],
            12: ["费用问题", "乱扣费", "多扣费", "费用异常"],
            13: ["业务办理问题", "办理失败", "办理不了", "办理困难"],
            14: ["服务问题", "服务态度", "客服", "投诉服务"],
            15: ["变更", "修改", "更改", "调整"],
            16: ["服务渠道", "营业厅", "网点", "办理地点"],
            17: ["规定不满", "不合理", "抗议", "意见"],
            18: ["营销问题", "推销", "骚扰", "强制营销"],
            19: ["网络问题", "信号", "网速", "断网", "连不上"],
            20: ["停复机", "停机", "复机", "暂停", "恢复"],
            21: ["信息安全", "隐私", "泄露", "安全"],
            22: ["重置", "修改", "补发", "找回"],
            23: ["使用方式", "怎么用", "如何使用", "操作方法"],
            24: ["号码状态", "号码查询", "状态查询"],
            25: ["工单处理", "工单结果", "处理进度"],
            26: ["打印", "邮寄", "发送", "寄送"],
            27: ["宽带覆盖", "覆盖范围", "能否安装"],
            28: ["移机", "装机", "拆机", "安装"],
            29: ["缴费", "交费", "付费", "充值"],
            30: ["下载", "设置", "配置", "安装"],
            31: ["补换卡", "换卡", "补卡", "重新办卡"],
            32: ["销户", "重开", "注销", "重新开户"],
            33: ["电商货品", "商品", "购买", "商城"]
        }

    def analyze_class_distribution(self, csv_file: str) -> Dict[int, int]:
        """分析类别分布"""
        df = pd.read_csv(csv_file)
        class_counts = Counter(df['c_numerical'])
        
        print("类别分布分析:")
        print("-" * 50)
        for class_id in sorted(class_counts.keys()):
            class_name = self.business_types.get(class_id, f"未知类型{class_id}")
            count = class_counts[class_id]
            percentage = (count / len(df)) * 100
            print(f"类别 {class_id:2d}: {class_name:<20} | {count:4d} 样本 ({percentage:5.2f}%)")
        
        return dict(class_counts)

    def identify_minority_classes(self, class_counts: Dict[int, int], 
                                threshold_ratio: float = 0.5) -> List[int]:
        """识别少数类别"""
        total_samples = sum(class_counts.values())
        avg_samples = total_samples / len(class_counts)
        threshold = avg_samples * threshold_ratio
        
        minority_classes = [
            class_id for class_id, count in class_counts.items() 
            if count < threshold
        ]
        
        print(f"\n识别到 {len(minority_classes)} 个少数类别 (样本数 < {threshold:.1f}):")
        for class_id in minority_classes:
            print(f"  类别 {class_id}: {self.business_types[class_id]} ({class_counts[class_id]} 样本)")
        
        return minority_classes

    def create_generation_prompts(self, class_id: int, examples: List[str]) -> str:
        """为特定类别创建生成提示词"""
        class_name = self.business_types[class_id]
        keywords = self.category_keywords.get(class_id, [])
        
        prompt = f"""你是中国移动客服对话数据生成专家。请根据以下要求生成客服对话数据。

任务：生成属于"{class_name}"类别的客服对话
关键词参考：{', '.join(keywords)}

现有样本示例：
"""
        
        for i, example in enumerate(examples[:3], 1):
            prompt += f"\n示例{i}：{example}\n"
        
        prompt += f"""
生成要求：
1. 生成5个新的客服对话，每个对话应该明确属于"{class_name}"类别
2. 对话格式：多轮对话用[SEP]分隔，通常以"您好请讲"或类似开场白开始
3. 对话内容要真实自然，符合中国移动客服场景
4. 确保生成的对话与给定类别高度相关
5. 避免与示例过于相似，保持多样性
6. 每个对话长度适中（100-300字）

请按以下格式输出：
对话1：[生成的对话内容]
对话2：[生成的对话内容]  
对话3：[生成的对话内容]
对话4：[生成的对话内容]
对话5：[生成的对话内容]
"""
        return prompt

    def call_llm_api(self, prompt: str, api_url: str = None, 
                     api_key: str = None) -> str:
        """调用LLM API生成样本"""
        # 这里可以接入不同的LLM API
        # 示例使用OpenAI API格式，你可以根据实际情况修改
        
        if not api_url:
            # 使用本地GLM-4服务（假设在8000端口运行）
            api_url = "http://localhost:8000/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        data = {
            "model": "glm-4",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,  # Slightly lower for more focused responses
            "max_tokens": 1500,  # Reduced to prevent overly long responses
            "top_p": 0.9,        # Add nucleus sampling for better performance
        }
        
        try:
            # Use shorter timeout with proper retry handling
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            print(f"API调用超时 (30秒)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return None
        except Exception as e:
            print(f"API调用失败: {e}")
            return None

    def parse_generated_conversations(self, generated_text: str) -> List[str]:
        """解析生成的对话"""
        conversations = []
        lines = generated_text.strip().split('\n')
        
        current_conversation = ""
        for line in lines:
            line = line.strip()
            if line.startswith('对话') and '：' in line:
                if current_conversation:
                    conversations.append(current_conversation.strip())
                current_conversation = line.split('：', 1)[1]
            elif current_conversation and line:
                current_conversation += " " + line
        
        if current_conversation:
            conversations.append(current_conversation.strip())
        
        return conversations

    def generate_synthetic_samples(self, csv_file: str, 
                                 minority_classes: List[int],
                                 target_samples_per_class: int = 100,
                                 api_url: str = None,
                                 api_key: str = None) -> List[Dict]:
        """为少数类别生成合成样本"""
        df = pd.read_csv(csv_file)
        synthetic_samples = []
        
        for class_id in minority_classes:
            print(f"\n为类别 {class_id} ({self.business_types[class_id]}) 生成合成样本...")
            
            # 获取该类别的现有样本
            class_samples = df[df['c_numerical'] == class_id]['sentence_sep'].tolist()
            current_count = len(class_samples)
            
            if current_count >= target_samples_per_class:
                print(f"  类别 {class_id} 已有 {current_count} 个样本，跳过")
                continue
            
            needed_samples = target_samples_per_class - current_count
            print(f"  需要生成 {needed_samples} 个样本")
            
            # 分批生成
            batch_size = 5  # 每次生成5个样本
            batches_needed = (needed_samples + batch_size - 1) // batch_size
            
            for batch in range(batches_needed):
                print(f"  正在生成批次 {batch + 1}/{batches_needed}...")
                
                # 创建生成提示词
                examples = random.sample(class_samples, min(3, len(class_samples)))
                prompt = self.create_generation_prompts(class_id, examples)
                
                # 调用LLM生成
                generated_text = self.call_llm_api(prompt, api_url, api_key)
                
                if generated_text:
                    # 解析生成的对话
                    new_conversations = self.parse_generated_conversations(generated_text)
                    
                    # 添加到合成样本列表
                    for conv in new_conversations:
                        if len(synthetic_samples) < needed_samples:
                            synthetic_samples.append({
                                'sentence_sep': conv,
                                'c_numerical': class_id,
                                'label_raw': self.business_types[class_id],
                                'synthetic': True
                            })
                    
                    print(f"    成功生成 {len(new_conversations)} 个样本")
                else:
                    print(f"    批次 {batch + 1} 生成失败")
                
                # 避免API调用过于频繁
                time.sleep(1)
        
        return synthetic_samples

    def create_balanced_dataset(self, original_csv: str, 
                              output_csv: str,
                              target_samples_per_class: int = 100,
                              api_url: str = None,
                              api_key: str = None):
        """创建平衡的数据集"""
        print("开始创建平衡数据集...")
        
        # 1. 分析当前分布
        class_counts = self.analyze_class_distribution(original_csv)
        
        # 2. 识别少数类别
        minority_classes = self.identify_minority_classes(class_counts, threshold_ratio=0.3)
        
        # 3. 生成合成样本
        synthetic_samples = self.generate_synthetic_samples(
            original_csv, minority_classes, target_samples_per_class, api_url, api_key
        )
        
        # 4. 合并原始数据和合成数据
        original_df = pd.read_csv(original_csv)
        if synthetic_samples:
            synthetic_df = pd.DataFrame(synthetic_samples)
            balanced_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        else:
            balanced_df = original_df
            print("未生成任何合成样本，使用原始数据")
        
        # 5. 保存平衡后的数据集
        balanced_df.to_csv(output_csv, index=False)
        
        print(f"\n数据集平衡完成！")
        print(f"原始样本数: {len(original_df)}")
        print(f"合成样本数: {len(synthetic_samples)}")
        print(f"总样本数: {len(balanced_df)}")
        print(f"输出文件: {output_csv}")
        
        # 6. 分析平衡后的分布
        print("\n平衡后的类别分布:")
        final_counts = balanced_df['c_numerical'].value_counts().sort_index()
        for class_id in sorted(final_counts.index):
            class_name = self.business_types.get(class_id, f"未知类型{class_id}")
            count = final_counts[class_id]
            print(f"类别 {class_id:2d}: {class_name:<20} | {count:4d} 样本")

def main():
    """主函数"""
    augmentor = CMCCDataAugmentation()
    
    # 配置参数
    original_train_file = "train_new.csv"
    balanced_train_file = "train_balanced.csv" 
    target_samples = 100  # 每个类别目标样本数
    
    # API配置（根据你的实际情况修改）
    api_url = "http://localhost:8000/v1/chat/completions"  # 本地GLM-4服务
    api_key = None  # 如果需要API key，在这里设置
    
    # 创建平衡数据集
    augmentor.create_balanced_dataset(
        original_csv=original_train_file,
        output_csv=balanced_train_file,
        target_samples_per_class=target_samples,
        api_url=api_url,
        api_key=api_key
    )

if __name__ == "__main__":
    main()