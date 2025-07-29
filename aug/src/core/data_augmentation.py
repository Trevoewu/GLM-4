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

    def get_business_category(self, class_id: int) -> float:
        """获取业务大类"""
        if class_id in [0, 2, 3, 4, 6, 8, 10, 11, 16, 23, 24, 25, 27, 33]:  # 咨询类
            return 0.0
        elif class_id in [5, 9, 12, 13, 14, 17, 18, 19, 21]:  # 投诉类
            return 1.0
        elif class_id in [1, 7, 15, 20, 22, 26, 28, 29, 30, 31, 32]:  # 办理类
            return 2.0
        else:
            return 0.0

    def create_generation_prompts(self, class_id: int, examples: List[str], batch_size: int = 3) -> str:
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
1. 生成{batch_size}个新的客服对话，每个对话应该明确属于"{class_name}"类别
2. 对话格式：多轮对话用[SEP]分隔，通常以"您好请讲"或类似开场白开始
3. 对话内容要真实自然，符合中国移动客服场景
4. 确保生成的对话与给定类别高度相关
5. 避免与示例过于相似，保持多样性
6. 每个对话长度适中（100-400字）

请按以下格式输出：
对话1：[生成的对话内容]
对话2：[生成的对话内容]  
对话3：[生成的对话内容]
对话4：[生成的对话内容]
对话5：[生成的对话内容]
{f'对话6：[生成的对话内容]' if batch_size > 5 else ''}
{f'对话7：[生成的对话内容]' if batch_size > 6 else ''}
{f'对话8：[生成的对话内容]' if batch_size > 7 else ''}
{f'对话9：[生成的对话内容]' if batch_size > 8 else ''}
{f'对话10：[生成的对话内容]' if batch_size > 9 else ''}
"""
        return prompt

    def call_llm_api(self, prompt: str, api_url: str = None, 
                     api_key: str = None, llm_config: dict = None) -> str:
        """调用LLM API生成样本"""
        # 使用配置文件中的参数，而不是硬编码
        if not llm_config:
            llm_config = {}
        
        if not api_url:
            api_url = llm_config.get('api_url', "http://localhost:8001/v1/chat/completions")
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # 从配置文件读取所有LLM参数
        data = {
            "model": llm_config.get('model_name', "glm-4"),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": llm_config.get('temperature', 0.8),
            "max_tokens": llm_config.get('max_tokens', 1200),
            "top_p": llm_config.get('top_p', 0.9),
            "repetition_penalty": llm_config.get('repetition_penalty', 1.1),
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
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
                                 target_samples_per_class: int = 200,
                                 batch_size: int = 10,
                                 max_retries: int = 5,
                                 api_url: str = None,
                                 api_key: str = None,
                                 class_specific_targets: Dict[int, int] = None,
                                 llm_config: dict = None) -> List[Dict]:
        """为少数类别生成合成样本（改进版）"""
        df = pd.read_csv(csv_file)
        synthetic_samples = []
        
        for class_id in minority_classes:
            print(f"\n为类别 {class_id} ({self.business_types[class_id]}) 生成合成样本...")
            
            # 获取该类别的现有样本
            class_samples = df[df['c_numerical'] == class_id]['sentence_sep'].tolist()
            current_count = len(class_samples)
            
            # 使用类别特定目标或默认目标
            target_count = class_specific_targets.get(class_id, target_samples_per_class) if class_specific_targets else target_samples_per_class
            
            if current_count >= target_count:
                print(f"  类别 {class_id} 已有 {current_count} 个样本，目标 {target_count}，跳过")
                continue
            
            needed_samples = target_count - current_count
            print(f"  当前样本数: {current_count}, 目标: {target_count}, 需要生成: {needed_samples}")
            
            # 分批生成（使用配置的batch_size）
            batches_needed = (needed_samples + batch_size - 1) // batch_size
            generated_count = 0
            
            for batch in range(batches_needed):
                print(f"  正在生成批次 {batch + 1}/{batches_needed}...")
                
                # 创建生成提示词
                examples = random.sample(class_samples, min(3, len(class_samples)))
                prompt = self.create_generation_prompts(class_id, examples, batch_size)
                
                # 调用LLM生成（带重试机制）
                retry_count = 0
                generated_text = None
                
                while retry_count < max_retries and generated_text is None:
                    try:
                        generated_text = self.call_llm_api(prompt, api_url, api_key, llm_config)
                        if generated_text:
                            break
                    except Exception as e:
                        print(f"    API调用失败 (第{retry_count + 1}次): {e}")
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"    等待 {retry_count * 2} 秒后重试...")
                        time.sleep(retry_count * 2)  # 递增等待时间
                
                if generated_text:
                    # 解析生成的对话
                    new_conversations = self.parse_generated_conversations(generated_text)
                    
                    # 添加到合成样本列表
                    batch_generated = 0
                    for conv in new_conversations:
                        if generated_count < needed_samples:
                            synthetic_samples.append({
                                'word_mf2': '',  # 添加缺失字段
                                'sentence_sep': conv,
                                'c_numerical': class_id,
                                'num_cnum': self.get_business_category(class_id),  # 添加业务大类
                                'label_raw': self.business_types[class_id],
                                'synthetic': True
                            })
                            generated_count += 1
                            batch_generated += 1
                    
                    print(f"    成功生成 {batch_generated} 个样本 (累计: {generated_count}/{needed_samples})")
                else:
                    print(f"    批次 {batch + 1} 生成失败，跳过")
                
                # 控制API调用频率
                time.sleep(0.5)
                
                # 如果已达到目标数量，提前结束
                if generated_count >= needed_samples:
                    print(f"  类别 {class_id} 已达到目标样本数 {target_count}")
                    break
        
        return synthetic_samples

    def create_balanced_dataset(self, original_csv: str, 
                              output_csv: str,
                              target_samples_per_class: int = 200,
                              batch_size: int = 10,
                              max_retries: int = 5,
                              api_url: str = None,
                              api_key: str = None,
                              class_specific_config: Dict = None,
                              llm_config: dict = None):
        """创建平衡的数据集"""
        print("开始创建平衡数据集...")
        
        # 1. 分析当前分布
        class_counts = self.analyze_class_distribution(original_csv)
        
        # 2. 识别少数类别
        minority_classes = self.identify_minority_classes(class_counts, threshold_ratio=0.1)
        
        # 3. 处理类别特定配置
        class_specific_targets = {}
        if class_specific_config:
            ultra_minority = class_specific_config.get('ultra_minority', {})
            minority = class_specific_config.get('minority', {})
            
            # 超少数类别目标
            if 'classes' in ultra_minority and 'target_samples' in ultra_minority:
                for class_id in ultra_minority['classes']:
                    class_specific_targets[class_id] = ultra_minority['target_samples']
            
            # 少数类别目标
            if 'classes' in minority and 'target_samples' in minority:
                for class_id in minority['classes']:
                    if class_id not in class_specific_targets:  # 超少数类别优先
                        class_specific_targets[class_id] = minority['target_samples']
        
        print(f"类别特定目标: {class_specific_targets}")
        
        # 4. 生成合成样本
        synthetic_samples = self.generate_synthetic_samples(
            original_csv, minority_classes, target_samples_per_class, 
            batch_size, max_retries, api_url, api_key, class_specific_targets, llm_config
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
    
    # 现在所有配置都在 augment_config.yaml 文件中
    # 请使用 run_augmentation.py 或 src/scripts/run_aug.py 来运行增强
    print("⚠️  请使用配置文件运行增强系统:")
    print("   python run_augmentation.py")
    print("   或: python src/scripts/run_aug.py")
    print("   所有配置在 augment_config.yaml 中管理")

if __name__ == "__main__":
    main()