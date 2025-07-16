import pandas as pd
import json
import os

def convert_to_glm4_format(csv_file, output_file, prompt_template):
    """
    将CMCC-34数据集转换为GLM-4微调格式
    
    Args:
        csv_file: 输入的CSV文件路径
        output_file: 输出的JSON文件路径
        prompt_template: 使用的prompt模板
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 业务类型映射
    business_types = {
        0: "咨询业务规定", 1: "办理取消", 2: "咨询业务资费", 3: "咨询营销活动", 4: "咨询办理方式",
        5: "投诉业务使用", 6: "咨询账户信息", 7: "办理开通", 8: "咨询业务订购", 9: "投诉不知情定制",
        10: "咨询产品功能", 11: "咨询用户资料", 12: "投诉费用", 13: "投诉业务办理", 14: "投诉服务",
        15: "办理变更", 16: "咨询服务渠道", 17: "投诉业务规定", 18: "投诉营销", 19: "投诉网络",
        20: "办理停复机", 21: "投诉信息安全", 22: "办理重置修改", 23: "咨询使用方式", 24: "咨询号码状态",
        25: "咨询工单处理", 26: "办理打印邮寄", 27: "咨询宽带覆盖", 28: "办理移机装机", 29: "办理缴费",
        30: "办理下载设置", 31: "办理补换卡", 32: "办理销户重开", 33: "咨询电商货品"
    }
    
    converted_data = []
    
    for idx, row in df.iterrows():
        # 获取对话内容
        conversation = row['sentence_sep']
        
        # 获取业务类型
        business_type_id = row['c_numerical']
        business_type_name = business_types.get(business_type_id, f"未知类型{business_type_id}")
        
        # 构建用户输入（使用prompt模板）
        user_input = prompt_template.format(conversation=conversation)
        
        # 构建助手回复
        assistant_response = f"理由：根据对话内容分析，用户的主要意图是{business_type_name}。\n意图：{business_type_name}"
        
        # 创建GLM-4格式的数据
        glm4_data = {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
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
    # 定义prompt模板
    prompt_template = """你是客服意图识别专家。根据对话内容判断用户意图。

对话内容：{conversation}

业务类型：
0:咨询业务规定 1:办理取消 2:咨询业务资费 3:咨询营销活动 4:咨询办理方式
5:投诉业务使用 6:咨询账户信息 7:办理开通 8:咨询业务订购 9:投诉不知情定制
10:咨询产品功能 11:咨询用户资料 12:投诉费用 13:投诉业务办理 14:投诉服务
15:办理变更 16:咨询服务渠道 17:投诉业务规定 18:投诉营销 19:投诉网络
20:办理停复机 21:投诉信息安全 22:办理重置修改 23:咨询使用方式 24:咨询号码状态
25:咨询工单处理 26:办理打印邮寄 27:咨询宽带覆盖 28:办理移机装机 29:办理缴费
30:办理下载设置 31:办理补换卡 32:办理销户重开 33:咨询电商货品

输出格式：
理由：[1句话说明]
意图：[选择最合适的意图]"""
    
    # 转换训练集
    convert_to_glm4_format(
        'train_new.csv', 
        'train.jsonl', 
        prompt_template
    )
    
    # 转换验证集
    convert_to_glm4_format(
        'dev_new.csv', 
        'dev.jsonl', 
        prompt_template
    )
    
    # 转换测试集
    convert_to_glm4_format(
        'test_new.csv', 
        'test.jsonl', 
        prompt_template
    ) 