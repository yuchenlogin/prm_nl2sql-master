# DeepSeek SQL 适配模块使用示例

## 快速开始

### 1. 基本使用

```python
from deepseek_sql import DeepSeekNL2SQL

# 初始化系统
deepseek = DeepSeekNL2SQL(
    model_name="Qwen/Qwen3-1.7B",
    pool_dir="./outputs/sql_proof_pool"
)

# 处理查询
result = deepseek.process_query(
    query="今年的总线索量是多少？",
    schema="CREATE TABLE dwd_ai_dc_sale_retail_leads_net_df(...)",
    knowledge="净新增线索量使用count(distinct customer_account_id)计算...",
    examples="示例：SELECT COUNT(DISTINCT customer_account_id) FROM table WHERE..."
)

print(f"最佳SQL: {result['best_sql']}")
print(f"过程奖励: {result['process_reward']['total_process_reward']}")
print(f"处理轮次: {result['total_rounds']}")
```

### 2. 批量处理数据集

```python
import json
from deepseek_sql import DeepSeekNL2SQL

# 初始化
deepseek = DeepSeekNL2SQL()

# 加载数据集
with open('data/nl2_sql_test.json', 'r') as f:
    dataset = json.load(f)

# 批量处理
results = []
for i, sample in enumerate(dataset[:10]):  # 示例：只处理前10个
    query = sample['query']
    # 解析schema等信息
    schema, knowledge, examples = parse_response(sample['response'])

    result = deepseek.process_query(
        query=query,
        schema=schema,
        knowledge=knowledge,
        examples=examples,
        problem_idx=f"demo_{i}"
    )

    results.append(result)
    print(f"Sample {i+1}: SQL generated, score = {result['best_score']:.3f}")

# 查看统计
print(f"处理了{len(results)}个样本")
avg_score = sum(r['best_score'] for r in results) / len(results)
print(f"平均分数: {avg_score:.3f}")
```

### 3. 与现有训练框架集成

```bash
# 使用DeepSeek GRPO训练
python training/train_deepseek_grpo.py --config config.yaml

# 运行完整评估
python evaluation/evaluator.py --model ./outputs/deepseek_checkpoints/best_model --test_file data/test.json --use_deepseek True
```

## 核心架构说明

### 三层验证架构

1. **SQL生成 + 自评估**
   - 生成SQL查询和推理过程
   - 模型自检语法和逻辑
   - 输出：`<thinking>...</thinking><sql>...</sql><self_eval>...</self_eval>`

2. **独立SQL验证器**
   - 语法验证：基于sqlparse
   - 逻辑验证：11种问题检测规则
   - 业务验证：时间范围、指标定义检查
   - 评分：0.0-1.0，支持多验证器

3. **元验证器**
   - 评估验证器质量
   - 检查一致性和可靠性
   - 提供置信度评估

### 过程奖励机制

```python
from deepseek_sql.reward_calculator import SQLProcessRewardCalculator

# 计算过程奖励
calculator = SQLProcessRewardCalculator()
process_reward = calculator.calculate_process_reward(
    problem_idx="example_01",
    query="查询问题",
    training_history=[
        {
            'round_number': 1,
            'generation': {...},
            'verifications': [...],
            'meta_verifications': [...]
        },
        # 更多轮次...
    ]
)

print(f"生成奖励: {process_reward['reward_breakdown']['generation']['contribution']:.3f}")
print(f"验证奖励: {process_reward['reward_breakdown']['verification']['contribution']:.3f}")
print(f"元验证奖励: {process_reward['reward_breakdown']['meta_verification']['contribution']:.3f}")
print(f"迭代奖励: {process_reward['reward_breakdown']['iteration']['contribution']:.3f}")
```

### 证明池管理

```python
from deepseek_sql.proof_pool import SQLProofPool

# 初始化证明池
pool = SQLProofPool("./outputs/sql_proof_pool")

# 添加证明
proof_id = pool.add_proof("problem_01", {
    'sql': "SELECT COUNT(*) FROM table",
    'thinking': "推理过程...",
    'self_eval': "自评估...",
    'verifications': [...],
    'meta_verifications': [...],
    'round_number': 1
})

# 获取最佳证明
best_proofs = pool.get_best_proofs("problem_01", count=3)
for i, proof in enumerate(best_proofs):
    print(f"第{i+1}名: {proof['final_score']:.3f} - {proof['sql']}")

# 获取统计
stats = pool.get_pool_stats("problem_01")
print(f"池大小: {stats['pool_size']}")
print(f"平均分数: {stats['score_stats']['mean']:.3f}")
```

## 配置示例

### DeepSeek GRPO 训练配置 (`config.yaml`)

```yaml
# 模型配置
model_name: "Qwen/Qwen3-1.7B"
torch_dtype: "bfloat16"
load_in_4bit: false
load_in_8bit: false

# 数据配置
train_file: "./data/nl2_sql_train.json"
test_file: "./data/nl2_sql_test.json"
val_split: 0.1

# DeepSeek配置
deepseek_max_rounds: 3
deepseek_n_generations_per_round: 2
deepseek_n_verifications_per_generation: 2
deepseek_process_reward_weight: 0.7
deepseek_final_reward_weight: 0.3

# 训练配置（调整以适应DeepSeek开销）
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5e-6
num_train_epochs: 3

# 输出配置
output_dir: "./outputs/deepseek_checkpoints"
wandb_project: "qwen3-nl2sql-deepseek-grpo"
```

## 性能优化建议

### 1. 内存优化
```python
# 使用梯度累积来减少内存占用
gradient_accumulation_steps: 4

# 使用量化（如果GPU内存不足）
load_in_8bit: true

# 减少每轮生成的数量
deepseek_n_generations_per_round: 1
```

### 2. 速度优化
```python
# 减少最大轮次
deepseek_max_rounds: 2

# 减少验证次数
deepseek_n_verifications_per_generation: 1

# 使用较低精度
torch_dtype: "float16"
```

### 3. 质量优化
```python
# 增加轮次以获得更好质量
deepseek_max_rounds: 4

# 增加验证器数量
deepseek_n_verifications_per_generation: 3

# 使用更高精度
torch_dtype: "bfloat16"
```

## 监控和调试

### 1. 查看证明池状态
```python
# 获取全局统计
stats = deepseek.get_pool_statistics()
print(f"总问题数: {stats['total_problems']}")
print(f"总证明数: {stats['total_proofs']}")
print(f"平均最佳分数: {stats['avg_best_score']:.3f}")
```

### 2. 分析过程奖励
```python
# 查看奖励分解
reward_breakdown = result['process_reward']['reward_breakdown']
for category, data in reward_breakdown.items():
    print(f"{category}: {data['score']:.3f} × {data['weight']:.2f} = {data['contribution']:.3f}")
```

### 3. 调试单轮结果
```python
# 查看训练历史
for i, round_data in enumerate(result['training_history']):
    print(f"\n第{i+1}轮:")
    print(f"  生成成功: {round_data['generation']['success']}")
    print(f"  SQL: {round_data['generation']['sql']}")
    print(f"  验证数量: {len(round_data['verifications'])}")
    if round_data['verifications']:
        print(f"  平均验证分数: {sum(v['score'] for v in round_data['verifications']) / len(round_data['verifications']):.3f}")
```

## 与现有代码的兼容性

### 1. 数据格式兼容
DeepSeek模块完全支持你现有的数据格式，自动解析：
- query: 用户问题
- response: 包含schema、knowledge、examples的完整响应
- type: 任务类型标记

### 2. 模型兼容
可以加载任何与Transformers兼容的模型：
```python
DeepSeekNL2SQL("Qwen/Qwen3-1.7B")           # 你现有的模型
DeepSeekNL2SQL("meta-llama/Llama-2-7b-hf")   # 其他模型
DeepSeekNL2SQL("/path/to/local/model")       # 本地模型
```

### 3. 训练集成
DeepSeek GRPO训练脚本可以与你现有的训练流程并行运行：
```bash
# 原始训练
python training/train_grpo.py --config config.yaml

# DeepSeek增强训练
python training/train_deepseek_grpo.py --config config.yaml
```

## 下一步

1. **运行示例测试**：先在小数据集上测试功能
2. **调整配置**：根据硬件条件优化性能参数
3. **对比实验**：比较原始方法 vs DeepSeek方法的效果
4. **参数调优**：根据奖励分解调整权重配置
5. **结果分析**：深入分析过程奖励与最终结果的相关性