# Qwen3 NL2SQL 过程奖励微调

这是一个完整的过程奖励模型（Process Reward Model, PRM）实现，基于DeepSeek-Math-V2的方法，应用于NL2SQL（自然语言到SQL生成）任务，使用**Qwen3-1.7B**模型和GRPO（Group Relative Policy Optimization）训练算法。

## 项目概述

### 主要特性

- **过程奖励方法**：不仅评估最终SQL的质量，还评估推理过程的质量
- **4维度奖励系统**：
  - 类型预测准确度（20%）：简单SQL vs 多步推理
  - 推理过程质量（25%）：<think>部分的逻辑完整性
  - 自我评估准确度（25%）：模型识别自身SQL问题的能力
  - SQL结构质量（30%）：SQL的有效性和完整性

- **完整的GRPO训练管道**：
  - 支持8卡A100-SXM4-80GB GPU分布式训练
  - 集成Weights & Biases (W&B)实验追踪
  - 自动检查点管理和最佳模型保存

- **11种常见SQL问题检测**：
  - 缺少WHERE条件
  - 缺少时间范围（NL2SQL特有）
  - JOIN条件不完整
  - GROUP BY与聚合函数不匹配
  - 括号不匹配
  - 日期格式错误
  - 等等

- **全面的评估指标**：
  - 类型分类准确率
  - SQL有效性评分
  - 推理过程质量
  - 自我评估准确率
  - 问题检测率
  - 复杂度分布

### 项目结构

```
qwen3_nl2sql_grpo/
├── config.yaml                  # 主配置文件（8GPU A100优化）
├── requirements.txt             # 依赖包（2025最新版本）
├── data/
│   └── data_loader.py          # 数据加载和处理模块
├── classifiers/
│   ├── complexity_classifier.py # SQL复杂度分类和问题检测
│   └── meta_classifier.py       # 分类结果验证
├── generator/
│   ├── prompts.py              # 提示词模板库
│   └── sql_generator.py        # SQL生成和推理
├── reward/
│   └── reward_model.py         # 过程奖励模型（4维度）
├── training/
│   ├── train_grpo.py           # GRPO主训练脚本
│   └── train_utils.py          # 日志、检查点、W&B工具
├── evaluation/
│   ├── evaluator.py            # 评估管道
│   └── metrics.py              # 指标计算
├── scripts/
│   └── prepare_data.py         # 数据验证和统计
├── README.md                    # 本文档
└── QUICK_START.md              # 快速开始指南
```

## 核心概念

### Qwen3模型的Thinking模式特性

**Qwen3-1.7B**是一个创新的模型，具备以下关键特性：

- **🧠 Thinking/Non-thinking模式无缝切换**：通过`enable_thinking=True/False`控制
- **🎯 原生思维推理能力**：模型自动生成`<thinking>...</thinking>`标签进行推理
- **🔄 实时模式切换**：可以在对话中动态切换推理模式
- **📊 32K超长上下文**：支持32768 tokens的上下文长度
- **⚡ 高效推理**：1.7B参数，适合资源受限环境

**与过程奖励模型的完美结合**：Qwen3的thinking模式天然适配我们的过程奖励方法，模型会自动生成详细的推理过程，我们可以直接评估这些推理过程的质量。

### 过程奖励模型 (PRM)

传统的强化学习方法使用单一的奖励信号（Outcome-based RM）来评估最终结果的质量。过程奖励模型则在生成过程的每一步都给出奖励信号，能够更细粒度地指导模型学习。

对于NL2SQL任务，我们实现了4维度的奖励系统：

1. **类型奖励** (type_reward, 20%)：
   - 评估模型是否正确分类了SQL的复杂度（简单 vs 多步推理）
   - 完全匹配：1.0，不匹配：0.0

2. **推理奖励** (thinking_reward, 25%)：
   - 评估<think>部分的质量
   - 考虑长度、关键词覆盖、逻辑连接词、结构化程度
   - 范围：0-1

3. **自我评估奖励** (self_assessment_reward, 25%)：
   - 评估模型是否能正确识别自己生成的SQL中的问题
   - 检查推理中是否提及了实际存在的问题
   - 范围：0-1

4. **SQL结构奖励** (sql_structure_reward, 30%)：
   - 评估生成的SQL的有效性和质量
   - 基于基本语法检查和问题严重程度
   - 范围：0-1

**总奖励** = 0.20 × type_reward + 0.25 × thinking_reward + 0.25 × self_assessment_reward + 0.30 × sql_structure_reward

### 复杂度分类

模型需要将SQL查询分类为两种类型之一：

- **"sql"**：简单SQL查询
  - 单表查询或简单的单次JOIN
  - 无复杂子查询
  - 聚合逻辑简单

- **"多步推理"**：复杂查询
  - 多表JOIN（2个或以上）
  - 使用CTE (WITH子句)
  - 包含UNION操作
  - 复杂的子查询和聚合逻辑

### SQL问题检测

系统自动检测11种常见的SQL问题：

1. `missing_where`：缺少WHERE条件 (严重度0.3)
2. `missing_time_range`：缺少时间范围 (严重度0.4) - NL2SQL特有
3. `incorrect_join`：JOIN缺少ON条件 (严重度0.5)
4. `missing_join_condition`：某些JOIN缺少ON条件 (严重度0.45)
5. `inconsistent_alias`：表别名不一致 (严重度0.25)
6. `missing_group_by`：聚合函数但无GROUP BY (严重度0.35)
7. `unclosed_parenthesis`：括号不匹配 (严重度0.8) - 严重错误
8. `empty_in_clause`：IN子句为空 (严重度0.7)
9. `invalid_date_format`：日期格式不正确 (严重度0.5)
10. `multiple_tables_no_join`：多表但未使用JOIN (严重度0.4)
11. `order_by_without_limit`：ORDER BY但无LIMIT (严重度0.2) - 轻微问题

## 安装

### 环境要求

- Python 3.10+
- CUDA 12.1+ （用于GPU训练）
- 8 × NVIDIA A100-SXM4-80GB GPU （推荐）

### 安装步骤

1. **克隆仓库**
```bash
cd /Users/yuch3n/qwen3_nl2sql_grpo
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
# venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置W&B**（可选但推荐）
```bash
wandb login
# 输入你的W&B API密钥
```

## 配置

### config.yaml 详解

主配置文件包含以下部分：

#### 模型配置
```yaml
model:
  name: "Qwen/Qwen3-1.7B"  # Qwen3模型，支持thinking模式
  torch_dtype: "bfloat16"  # A100原生支持bfloat16
  max_seq_length: 32768    # 32K上下文长度
```

#### 数据配置
```yaml
data:
  train_file: "./data/nl2_sql_cold_start_sft_all_train_swift_9501_1231.json"  # 9501样本
  test_file: "./data/nl2_sql_cold_start_sft_all_test_swift_830_1231.json"    # 830样本
  val_split: 0.1                     # 验证集比例
```

#### 训练配置 - 8GPU A100优化
```yaml
training:
  per_device_train_batch_size: 16    # Qwen3-1.7B可以设置更大批次
  gradient_accumulation_steps: 2     # 梯度累积步数
  # 有效批大小 = 8 GPU × 16 batch × 2 accumulation = 512

  learning_rate: 7.3e-6              # 微调学习率
  lr_scheduler_type: "cosine"        # 余弦退火
  warmup_steps: 100                  # 预热步数

  bf16: true                         # bfloat16混合精度
  tf32: true                         # TensorFloat-32加速

  optim: "adamw_8bit"                # 8位优化器节省显存
```

#### GRPO配置
```yaml
grpo:
  num_generations: 4                 # 每样本生成4个候选
  temperature: 0.7                   # 生成温度
  max_new_tokens: 1024               # 最大生成长度

  reward_weights:
    type_reward: 0.20                # 类型预测准确度
    thinking_reward: 0.25            # 推理过程质量
    self_assessment_reward: 0.25     # 自我评估准确度
    sql_structure_reward: 0.30       # SQL结构质量（最重）
```

#### W&B配置
```yaml
wandb:
  enabled: true                      # 启用实验追踪
  project: "qwen3-nl2sql-grpo"       # 项目名
  tags:
    - "nl2sql"
    - "process_reward"
    - "qwen3"
    - "grpo"
```

## 数据格式

### 输入数据格式

JSON格式，每条数据包含：

```json
{
  "query": "今年的总线索量是多少？",
  "response": "<think>\n问题要求查询今年的线索总量。\n...\n</think>\n\n<answer>\nSELECT COUNT(...) FROM ...\n</answer>",
  "type": "sql"
}
```

### 输出数据格式

训练和评估过程生成的标准化数据：

```python
{
  "query": "问题",
  "thinking": "推理过程",
  "sql": "SQL查询",
  "complexity_type": "sql" 或 "多步推理"
}
```

## 快速开始

### 1. 数据准备

```bash
python scripts/prepare_data.py \
  --train_file ./data/nl2_sql_cold_start_sft_all_train_swift_9501_1231.json \
  --test_file ./data/nl2_sql_cold_start_sft_all_test_swift_830_1231.json \
  --output_dir ./outputs
```

生成的报告位置：`./outputs/data_preparation_report.json`

### 2. 训练

```bash
# 基础训练
python training/train_grpo.py --config config.yaml

# 从检查点恢复训练
python training/train_grpo.py --config config.yaml --resume ./outputs/checkpoints/checkpoint-1000

# 自定义设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python training/train_grpo.py --config config.yaml
```

**预期输出**：
- 检查点：`./outputs/checkpoints/checkpoint-*/`
- 最佳模型：`./outputs/checkpoints/best_model/`
- 日志：`./outputs/logs/training.log`
- W&B仪表板：https://wandb.ai/your-entity/qwen3-nl2sql-grpo

### 3. 评估

```bash
python evaluation/evaluator.py \
  --model ./outputs/checkpoints/best_model \
  --test_file /path/to/test.json \
  --output ./outputs/evaluation_report.json
```

**预期输出**：`./outputs/evaluation_report.json` 包含：
- 汇总统计
- 各项指标
- 示例预测
- 错误分析

## 工作流程

### 完整训练工作流程

```bash
# 1. 准备环境
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2. 准备数据
python scripts/prepare_data.py \
  --train_file ./data/nl2_sql_cold_start_sft_all_train_swift_9501_1231.json \
  --test_file ./data/nl2_sql_cold_start_sft_all_test_swift_830_1231.json

# 3. 启动训练
python training/train_grpo.py --config config.yaml

# 4. 等待训练完成（预计2-3小时，取决于数据量）

# 5. 评估最佳模型
python evaluation/evaluator.py \
  --model ./outputs/checkpoints/best_model \
  --test_file ./data/nl2_sql_cold_start_sft_all_test_swift_830_1231.json

# 6. 查看结果
cat ./outputs/data_preparation_report.json
cat ./outputs/evaluation_report.json
```

## W&B集成

项目默认集成了Weights & Biases用于实验追踪。

### 配置W&B

1. **注册账号**：https://wandb.ai/

2. **获取API密钥**：访问 https://wandb.ai/settings/profile

3. **登录**：
```bash
wandb login
# 粘贴你的API密钥
```

4. **修改config.yaml**（可选）：
```yaml
wandb:
  entity: "your-username-or-team"  # 改为你的用户名或团队名
```

### W&B仪表板

训练时，实时指标会上传到W&B。访问：
```
https://wandb.ai/your-entity/qwen3-nl2sql-grpo
```

查看：
- 训练损失曲线
- 奖励分解（4维度分别显示）
- 学习率调度
- GPU显存使用情况
- 模型工件

## 代码模块详解

### 核心模块

#### 1. `data/data_loader.py` - 数据加载器
- `NL2SQLExample`：单样本封装
- `NL2SQLDataLoader`：主加载类
- `validate_data()`：数据验证函数

**关键功能**：
- 解析JSON格式数据
- 提取<think>和<answer>标签
- 缓存处理（pickle格式）
- 数据分割

#### 2. `classifiers/complexity_classifier.py` - 复杂度分类器
- `SQLComplexity`：枚举类（SIMPLE, MULTI_STEP）
- `ComplexityClassifier`：主分类类
- 11种问题检测规则

**关键功能**：
- 分类SQL复杂度
- 检测常见问题
- 计算严重程度

#### 3. `classifiers/meta_classifier.py` - 元分类器
- `MetaClassifier`：验证分类结果质量

**关键功能**：
- 验证复杂度分类的合理性
- 评估推理过程质量
- 计算置信度分数

#### 4. `generator/prompts.py` - 提示词模板
- `PromptTemplates`：模板类

**内容**：
- 系统提示词
- SQL生成提示词
- 少样本示例（2个复杂度案例）
- 业务知识背景
- 数据表结构

#### 5. `generator/sql_generator.py` - SQL生成器
- `SQLGenerator`：生成类

**关键功能**：
- 加载Qwen2.5模型
- 生成SQL和推理过程
- 自动复杂度分类
- 生成自我评估

#### 6. `reward/reward_model.py` - 奖励模型
- `ProcessRewardModel`：4维度奖励

**关键功能**：
- 计算type_reward（0.20）
- 计算thinking_reward（0.25）
- 计算self_assessment_reward（0.25）
- 计算sql_structure_reward（0.30）
- 批量计算和统计

#### 7. `training/train_grpo.py` - GRPO训练脚本
- `NL2SQLTrainer`：训练类
- `load_config()`：加载YAML配置

**关键功能**：
- 8GPU DDP分布式训练
- GRPO算法实现
- 检查点管理
- W&B集成
- 性能监控

#### 8. `training/train_utils.py` - 训练工具
- `WandBLogger`：W&B集成
- `CheckpointManager`：检查点管理
- `PerformanceMonitor`：性能监控
- `GPUMonitor`：GPU监控

#### 9. `evaluation/evaluator.py` - 评估管道
- `Evaluator`：评估类

**关键功能**：
- 加载微调模型
- 在测试集上推理
- 计算所有指标
- 生成详细报告

#### 10. `evaluation/metrics.py` - 评估指标
- `Metrics`：指标计算

**包含指标**（10+个）：
- 类型准确率
- SQL有效性
- 推理质量
- 自我评估准确率
- 问题检测率
- 覆盖率指标等

## 故障排查

### 常见问题

**Q1：CUDA内存不足**
```
RuntimeError: CUDA out of memory
```

解决方案：
- 减小`per_device_train_batch_size`（在config.yaml中）
- 增加`gradient_accumulation_steps`
- 使用`load_in_8bit: true`

**Q2：模型加载缓慢**
```
Downloading: ...
```

解决方案：
- 检查网络连接
- 使用本地模型路径
- 设置`HF_HOME`环境变量

**Q3：W&B连接失败**
```
wandb: offline
```

解决方案：
```bash
wandb online
wandb login
```

**Q4：数据加载出错**
```
ValueError: No valid data found
```

解决方案：
- 验证JSON格式
- 运行`python scripts/prepare_data.py`检查数据有效性
- 检查<think>和<answer>标签是否完整

## 预期性能

基于8 × A100-SXM4-80GB的预期性能：

| 指标 | 值 |
|-----|-----|
| 有效批大小 | 128 |
| 吞吐量 | ~128 样本/秒 |
| 单轮训练时间 | ~1.5 小时 |
| 3轮训练总时间 | ~4.5 小时 |
| 显存使用 | ~40GB（共320GB）|
| 最终模型大小 | ~3.5GB |

**注**：Qwen3-1.7B相比14B模型，训练速度提升约60%，显存占用减少约75%。

## 输出文件

训练后生成的文件结构：

```
outputs/
├── checkpoints/
│   ├── checkpoint-500/          # 中间检查点
│   ├── best_model/              # 最佳模型
│   └── final_model/             # 最终模型
├── logs/
│   ├── training.log             # 训练日志
│   └── data_preparation.log     # 数据日志
├── data_preparation_report.json # 数据统计
├── evaluation_report.json       # 评估结果
└── performance_report.txt       # 性能报告
```

## 扩展和定制

### 自定义奖励权重

修改`config.yaml`中的`reward_weights`部分：

```yaml
grpo:
  reward_weights:
    type_reward: 0.15              # 降低类型权重
    thinking_reward: 0.30          # 增加思考权重
    self_assessment_reward: 0.20
    sql_structure_reward: 0.35     # 最高权重
```

### 添加自定义问题检测

在`classifiers/complexity_classifier.py`的`ISSUE_RULES`中添加：

```python
ISSUE_RULES = {
    'your_issue': {
        'pattern': lambda sql: your_check(sql),
        'severity': 0.4,
        'description': '你的问题描述'
    },
    # ... 其他规则
}
```

### 自定义提示词模板

编辑`generator/prompts.py`中的`PromptTemplates`类。

## 论文参考

- DeepSeek-Math-V2：https://arxiv.org/abs/2405.03187
- GRPO：TRL库文档
- Qwen2.5：https://huggingface.co/Qwen

## 许可证

本项目仅供学习和研究使用。

## 常用命令速查表

```bash
# 启动训练
python training/train_grpo.py --config config.yaml

# 恢复训练
python training/train_grpo.py --config config.yaml --resume ./outputs/checkpoints/checkpoint-1000

# 评估模型
python evaluation/evaluator.py --model ./outputs/checkpoints/best_model --test_file <test_file>

# 准备数据
python scripts/prepare_data.py --train_file <train> --test_file <test>

# 查看日志
tail -f ./outputs/logs/training.log

# 查看W&B
wandb sync

# 清理缓存
rm -rf ./data/cache/*.pkl
```

## 支持和反馈

遇到问题或有建议？请提出Issue或联系开发者。

---

**最后更新**：2025年1月15日
**版本**：1.0.0
**兼容性**：PyTorch 2.9.1+, Transformers 4.57.5+, TRL 0.26.2+
