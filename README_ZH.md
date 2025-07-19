# GLM-4 微调用于 CMCC-34 意图分类

本仓库包含使用 QLoRA（量化低秩适应）在 CMCC-34 数据集上微调 GLM-4-9B 进行意图分类的完整流程。项目包括数据准备、模型训练、评估和推理功能。

## 🚀 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install torch transformers peft bitsandbytes accelerate
pip install scikit-learn matplotlib seaborn tqdm

# 支持 CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 数据准备

```bash
# 将原始 CSV 数据转换为 GLM-4 格式（使用系统提示）
cd finetune/data/cmcc-34
python convert_data.py

# 如需重新生成数据集
python regenerate_dataset.py
```

### 3. 模型训练

```bash
# 使用 QLoRA 训练模型
cd finetune
python train_cmcc34_system_prompt.py
```

### 4. 模型评估

```bash
# 快速评估（100个样本）
cd evaluation
python evaluate.py --quick --samples 100

# 完整评估
python evaluate.py 
```

### 5. 模型推理

```bash
# 交互式命令行界面
cd inference
python trans_cli_finetuned_demo.py

# Web 界面
python trans_web_finetuned_demo.py
```

## 📁 项目结构

```
GLM-4/
├── finetune/                    # 微调流程
│   ├── configs/                 # 训练配置
│   │   └── cmcc34_qlora_system_prompt.yaml
│   ├── data/                    # 数据处理脚本
│   │   └── cmcc-34/
│   │       ├── convert_data.py
│   │       ├── regenerate_dataset.py
│   │       ├── train.jsonl
│   │       └── test.jsonl
│   ├── output/                  # 训练输出
│   │   └── cmcc34_qlora_system_prompt/
│   └── train_cmcc34_system_prompt.py
├── evaluation/                  # 模型评估
│   ├── evaluate.py             # 主评估脚本
│   └── output/                 # 评估结果
├── inference/                   # 模型推理
│   ├── trans_cli_finetuned_demo.py
│   ├── trans_web_finetuned_demo.py
│   └── test_finetuned_model.py
├── demo/                        # 演示应用
├── resources/                   # 附加资源
└── README.md
```

## 🔧 配置

### 训练配置

训练使用 QLoRA，主要参数如下：

- **基础模型**: GLM-4-9B-0414
- **量化**: 4-bit (QLoRA)
- **LoRA 秩**: 64
- **LoRA Alpha**: 128
- **学习率**: 2e-4
- **批次大小**: 4
- **最大步数**: 5000
- **系统提示**: 针对意图分类优化

### 评估配置

- **测试数据集**: CMCC-34 测试集
- **指标**: 准确率、F1-宏平均、F1-加权平均
- **输出**: 失败预测、混淆矩阵、详细分析
- **重试逻辑**: 自动重试和内容截断

## 📊 模型性能

### 可用检查点

`finetune/output/cmcc34_qlora_system_prompt/` 中有多个检查点：

- `checkpoint-500/` - 早期训练
- `checkpoint-1000/` - 1000 步
- `checkpoint-2000/` - 2000 步
- `checkpoint-3000/` - 3000 步
- `checkpoint-4000/` - 4000 步
- `checkpoint-5000/` - 最新（推荐）

### 性能指标

模型达到：
- **准确率**: ~85-90%（取决于检查点）
- **F1-宏平均**: ~0.85-0.90
- **F1-加权平均**: ~0.85-0.90

## 🎯 意图分类

模型将用户意图分类为 34 个类别：

| 意图ID | 意图名称 |
|--------|----------|
| 0 | 咨询（含查询）业务规定 |
| 1 | 办理取消 |
| 2 | 咨询（含查询）业务资费 |
| 3 | 咨询（含查询）营销活动信息 |
| 4 | 咨询（含查询）办理方式 |
| 5 | 投诉（含抱怨）业务使用问题 |
| 6 | 咨询（含查询）账户信息 |
| 7 | 办理开通 |
| 8 | 咨询（含查询）业务订购信息查询 |
| 9 | 投诉（含抱怨）不知情定制问题 |
| 10 | 咨询（含查询）产品/业务功能 |
| ... | ... |

## 🛠️ 使用示例

### 命令行评估

```bash
# 快速评估 50 个样本
python evaluate.py --quick --samples 50

# 使用自定义模型路径进行完整评估
python evaluate.py --model-path ../finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000

# 自定义批次大小和输出目录
python evaluate.py --batch-size 25 --output-dir my_evaluation_results
```

### 编程使用

```python
from evaluation.evaluate import SystemPromptEvaluator

# 初始化评估器
evaluator = SystemPromptEvaluator(
    base_model_path="THUDM/GLM-4-9B-0414",
    finetuned_path="finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000",
    test_file="finetune/data/cmcc-34/test.jsonl",
    output_dir="evaluation_results"
)

# 加载模型并评估
evaluator.load_model()
test_data = evaluator.load_test_data()
results = evaluator.evaluate_batch(test_data)

# 打印结果
evaluator.print_results(results)
evaluator.save_final_results(results)
```

### 交互式推理

```bash
# 启动交互式命令行界面
cd inference
python trans_cli_finetuned_demo.py

# 示例对话：
# 用户: 我想查询我的余额
# 助手: 意图：6:咨询（含查询）账户信息
```

## 📈 评估结果

评估脚本生成全面的结果：

### 输出文件

- `failed_predictions.json` - 失败预测详情
- `error_predictions.json` - 错误预测分析
- `confusion_matrix.png` - 可视化混淆矩阵
- `detailed_analysis.json` - 每类性能指标
- `system_prompt_evaluation_results.json` - 完整结果

### 分析功能

- **失败预测分析**: 详细的错误分类
- **混淆矩阵**: 分类错误的可视化表示
- **每类性能**: 每个意图的精确率、召回率、F1分数
- **错误分布**: 最易混淆的意图对
- **重试逻辑**: 自动处理长输入和内存错误

## 🔍 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 减少批次大小
   python evaluate.py --batch-size 10
   
   # 使用较小的检查点
   python evaluate.py --model-path checkpoint-2000
   ```

2. **模型加载错误**
   ```bash
   # 检查模型路径
   ls finetune/output/cmcc34_qlora_system_prompt/
   
   # 验证基础模型
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('THUDM/GLM-4-9B-0414')"
   ```

3. **数据加载问题**
   ```bash
   # 重新生成数据集
   cd finetune/data/cmcc-34
   python regenerate_dataset.py
   ```

### 内存要求

- **训练**: ~24GB GPU 内存（使用 QLoRA）
- **评估**: ~10GB GPU 内存
- **推理**: ~10GB GPU 内存

## 🚀 高级用法

### 自定义训练

```bash
# 修改训练配置
vim finetune/configs/cmcc34_qlora_system_prompt.yaml

# 使用自定义参数训练
python train_cmcc34_system_prompt.py --config custom_config.yaml
```

### 自定义评估

```bash
# 在自定义测试集上评估
python evaluate.py --test-file custom_test.jsonl

# 比较多个检查点
for checkpoint in 1000 2000 3000 4000 5000; do
    python evaluate.py --model-path checkpoint-$checkpoint --output-dir eval_$checkpoint
done
```

### 生产部署

```python
# 为生产环境加载模型
from inference.model_loader import load_finetuned_model

model, tokenizer = load_finetuned_model(
    base_model_path="THUDM/GLM-4-9B-0414",
    finetuned_path="finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000"
)

# 批量推理
def classify_intents(texts):
    results = []
    for text in texts:
        intent = model.generate_intent(text)
        results.append(intent)
    return results
```

## 📚 参考文献

- [GLM-4 论文](https://arxiv.org/abs/2401.09602)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [CMCC-34 数据集](https://github.com/THUDM/GLM-4)
- [Transformers 文档](https://huggingface.co/docs/transformers)

## 🤝 贡献

1. Fork 本仓库
2. 创建功能分支
3. 进行修改
4. 如适用，添加测试
5. 提交拉取请求

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- THUDM 提供的 GLM-4 模型
- 微软提供的 QLoRA 技术
- 开源社区提供的各种工具和库