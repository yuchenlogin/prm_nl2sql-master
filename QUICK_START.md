# 快速开始指南 - 5分钟上手

## 最小化安装（3步）

### 步骤 1: 安装依赖（2分钟）
```bash
# 进入项目目录
cd /Users/yuch3n/qwen3_nl2sql_grpo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 准备数据（1分钟）
```bash
bash prepare_data.sh
```

### 步骤 3: 开始训练（2分钟）
```bash
# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 启动训练
python training/train_grpo.py --config config.yaml
```

**就这样！** 模型开始在8卡A100上训练。

---

## 一键命令

```bash
source venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
python training/train_grpo.py --config config.yaml
```

---

## 预期输出

训练时会看到：
```
2025-01-15 10:30:45 - training - INFO - 加载模型: Qwen/Qwen2.5-14B-Instruct
2025-01-15 10:30:50 - training - INFO - 模型已加载到: cuda
2025-01-15 10:30:51 - training - INFO - 加载了 9501 条训练数据
2025-01-15 10:30:52 - training - INFO - ==================================================
2025-01-15 10:30:52 - training - INFO - 开始GRPO训练
2025-01-15 10:30:52 - training - INFO - ==================================================
2025-01-15 10:30:52 - training - INFO - 训练开始...
[Progress bar showing training steps]
```

---

## 输出位置

| 内容 | 位置 |
|-----|-----|
| 检查点 | `./outputs/checkpoints/` |
| 最佳模型 | `./outputs/checkpoints/best_model/` |
| 训练日志 | `./outputs/logs/training.log` |
| W&B链接 | 控制台会打印 |

---

## 评估（训练完成后）

```bash
python evaluation/evaluator.py \
  --model ./outputs/checkpoints/best_model \
  --test_file ./data/nl2_sql_cold_start_sft_all_test_swift_830_1231.json
```

查看结果：
```bash
cat ./outputs/evaluation_report.json
```

---

## 主要配置参数

需要调整？编辑 `config.yaml`：

```yaml
# 训练轮数
training:
  num_train_epochs: 3

# 批大小（per GPU）
  per_device_train_batch_size: 8

# 学习率
  learning_rate: 7.3e-6

# 奖励权重
grpo:
  reward_weights:
    type_reward: 0.20
    thinking_reward: 0.25
    self_assessment_reward: 0.25
    sql_structure_reward: 0.30
```

---

## 常见问题速解

| 问题 | 解决方案 |
|-----|--------|
| CUDA内存不足 | 在config.yaml中改`per_device_train_batch_size: 4` |
| W&B离线 | 运行`wandb login` |
| 模型加载慢 | 检查网络，或使用本地路径 |
| 数据验证失败 | 检查JSON格式，运行`prepare_data.py` |

---

## 下一步

1. ✅ **训练完成**？评估最佳模型
2. 📊 **查看W&B仪表板**了解训练曲线
3. 📈 **对比指标**检查改进效果
4. 🎯 **分析错误**改进提示词

---

## 获取帮助

- 详细文档：见 `README.md`
- 查看日志：`tail -f ./outputs/logs/training.log`
- 检查数据：`python scripts/prepare_data.py`
- 调试模式：代码中加入 `logger.debug()`

---

## 训练时间预期

- **单卡（K80）**：~30小时
- **单卡（A100）**：~3小时
- **8卡A100（DDP）**：~45分钟

您的配置：**8 × A100-SXM4-80GB** → ~45分钟到2小时（取决于参数）

---

**准备好了吗？运行上面的3个步骤开始吧！** 🚀
