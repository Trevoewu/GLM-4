import pandas as pd
import json
import os

def convert_to_glm4_format(csv_file, output_file):
    """
    将CMCC-34数据集转换为GLM-4微调格式，使用system prompt优化
    
    Args:
        csv_file: 输入的CSV文件路径
        output_file: 输出的JSON文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 业务类型映射 - 更新为新的格式
    business_types = {
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
    
    # 系统提示词 - 包含所有业务类型信息
    system_prompt = """你是客服意图识别专家。分析对话内容，判断用户最终意图。

对话格式：多个说话轮次用[SEP]分隔，通常以"您好请讲"开始。

判断标准：
- 关注用户最终目标，不是中间过程
- 结合关键词和上下文综合判断

业务类型列表：
0:咨询（含查询）业务规定 1:办理取消 2:咨询（含查询）业务资费 3:咨询（含查询）营销活动信息 4:咨询（含查询）办理方式
5:投诉（含抱怨）业务使用问题 6:咨询（含查询）账户信息 7:办理开通 8:咨询（含查询）业务订购信息查询 9:投诉（含抱怨）不知情定制问题
10:咨询（含查询）产品/业务功能 11:咨询（含查询）用户资料 12:投诉（含抱怨）费用问题 13:投诉（含抱怨）业务办理问题 14:投诉（含抱怨）服务问题
15:办理变更 16:咨询（含查询）服务渠道信息 17:投诉（含抱怨）业务规定不满 18:投诉（含抱怨）营销问题 19:投诉（含抱怨）网络问题
20:办理停复机 21:投诉（含抱怨）信息安全问题 22:办理重置/修改/补发 23:咨询（含查询）使用方式 24:咨询（含查询）号码状态
25:咨询（含查询）工单处理结果 26:办理打印/邮寄 27:咨询（含查询）宽带覆盖范围 28:办理移机/装机/拆机 29:办理缴费
30:办理下载/设置 31:办理补换卡 32:办理销户/重开 33:咨询（含查询）电商货品信息

输出格式：
意图：[选择最合适的意图]"""
    
    converted_data = []
    
    for idx, row in df.iterrows():
        # 获取对话内容
        conversation = row['sentence_sep']
        
        # 获取业务类型
        business_type_id = row['c_numerical']
        business_type_name = business_types.get(business_type_id, f"未知类型{business_type_id}")
        
        # 构建助手回复 - 更新为新的格式
        assistant_response = f"意图：{business_type_id}：{business_type_name}"
        
        # 创建GLM-4格式的数据，使用system prompt
        glm4_data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"对话：{conversation}"
                },
                {
                    "role": "assistant", 
                    "content": assistant_response
                }
            ]
        }
        
        converted_data.append(glm4_data)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共处理 {len(converted_data)} 条数据")
    print(f"输出文件：{output_file}")

if __name__ == "__main__":
    # 转换训练集
    convert_to_glm4_format('train_new.csv', 'train.jsonl')
    
    # 转换验证集
    convert_to_glm4_format('dev_new.csv', 'dev.jsonl')
    
    # 转换测试集
    convert_to_glm4_format('test_new.csv', 'test.jsonl') 