#!/usr/bin/env python3
"""
运行CMCC-34数据集的LLM样本合成增强
Usage: python run_aug.py [--config CONFIG_FILE] [--dry-run]
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from ..core.data_augmentation import CMCCDataAugmentation

def setup_logging(log_file: str, log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_file: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        sys.exit(1)

def validate_config(config: dict) -> bool:
    """验证配置文件"""
    required_keys = [
        'augmentation',
        'llm_config', 
        'output'
    ]
    
    for key in required_keys:
        if key not in config:
            logging.error(f"配置文件缺少必需的键: {key}")
            return False
    
    return True

def check_prerequisites(config: dict) -> bool:
    """检查前置条件"""
    # 检查数据文件是否存在
    data_files = ['../data/cmcc-34/train_new.csv', '../data/cmcc-34/dev_new.csv']
    missing_files = []
    
    for file in data_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logging.error(f"缺少数据文件: {missing_files}")
        return False
    
    # 检查API是否可用（可选）
    api_url = config['llm_config']['api_url']
    if api_url and api_url.startswith('http'):
        try:
            import requests
            response = requests.get(api_url.replace('/v1/chat/completions', '/health'), timeout=5)
            if response.status_code != 200:
                logging.warning(f"API健康检查失败: {api_url}")
        except Exception as e:
            logging.warning(f"无法连接到API: {e}")
    
    return True

def run_augmentation(config: dict, dry_run: bool = False):
    """运行数据增强"""
    logging.info("开始数据增强过程...")
    
    # 创建增强器实例
    augmentor = CMCCDataAugmentation()
    
    # 从配置中获取参数
    aug_config = config['augmentation']
    llm_config = config['llm_config']
    output_config = config['output']
    
    if dry_run:
        logging.info("DRY RUN模式：仅分析数据分布，不生成新样本")
        
        # 只分析分布
        train_file = "../data/cmcc-34/train_new.csv"
        class_counts = augmentor.analyze_class_distribution(train_file)
        minority_classes = augmentor.identify_minority_classes(
            class_counts, 
            aug_config.get('minority_threshold_ratio', 0.3)
        )
        
        logging.info(f"识别到 {len(minority_classes)} 个少数类别")
        return
    
    # 运行完整的数据增强
    try:
        # 处理训练集
        logging.info("处理训练集...")
        augmentor.create_balanced_dataset(
            original_csv="../data/cmcc-34/train_new.csv",
            output_csv=output_config['balanced_train_file'],
            target_samples_per_class=aug_config['target_samples_per_class'],
            batch_size=aug_config.get('batch_size', 10),
            max_retries=aug_config.get('max_retries', 5),
            api_url=llm_config['api_url'],
            api_key=llm_config.get('api_key'),
            class_specific_config=config.get('class_specific_strategies', {})
        )
        
        # 处理验证集（如果需要）
        if os.path.exists("../finetune/data/cmcc-34/dev_new.csv") and output_config.get('balanced_dev_file'):
            logging.info("处理验证集...")
            augmentor.create_balanced_dataset(
                original_csv="../finetune/data/cmcc-34/dev_new.csv", 
                output_csv=output_config['balanced_dev_file'],
                target_samples_per_class=aug_config['target_samples_per_class'] // 5,  # 验证集样本较少
                api_url=llm_config['api_url'],
                api_key=llm_config.get('api_key')
            )
        
        logging.info("数据增强完成！")
        
    except Exception as e:
        logging.error(f"数据增强过程中出错: {e}")
        sys.exit(1)

def convert_to_jsonl(config: dict):
    """将增强后的CSV转换为JSONL格式用于训练"""
    logging.info("转换增强后的数据为JSONL格式...")
    
    try:
        from ..core.convert_data import convert_to_glm4_format
        
        output_config = config['output']
        
        # 转换训练集
        if os.path.exists(output_config['balanced_train_file']):
            convert_to_glm4_format(
                output_config['balanced_train_file'],
                'train_balanced.jsonl'
            )
            logging.info("训练集转换完成")
        
        # 转换验证集
        if os.path.exists(output_config.get('balanced_dev_file', '')):
            convert_to_glm4_format(
                output_config['balanced_dev_file'],
                'dev_balanced.jsonl'
            )
            logging.info("验证集转换完成")
            
    except Exception as e:
        logging.error(f"JSONL转换失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CMCC-34数据集LLM样本合成增强")
    parser.add_argument(
        "--config", 
        default="augment_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="仅分析数据分布，不实际生成样本"
    )
    parser.add_argument(
        "--convert-only",
        action="store_true", 
        help="仅转换现有的平衡数据为JSONL格式"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 验证配置
    if not validate_config(config):
        sys.exit(1)
    
    # 设置日志
    setup_logging(
        config['output'].get('log_file', 'augmentation.log'),
        config['output'].get('log_level', 'INFO')
    )
    
    logging.info("="*60)
    logging.info("CMCC-34数据集LLM样本合成增强")
    logging.info("="*60)
    
    # 检查前置条件
    if not check_prerequisites(config):
        sys.exit(1)
    
    if args.convert_only:
        # 仅转换模式
        convert_to_jsonl(config)
    else:
        # 运行数据增强
        run_augmentation(config, args.dry_run)
        
        # 转换为JSONL格式
        if not args.dry_run:
            convert_to_jsonl(config)
    
    logging.info("任务完成！")

if __name__ == "__main__":
    main()