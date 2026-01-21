# 安装和配置指南

## 系统要求

- Python 3.10+
- CUDA 12.1+ （用于GPU训练）
- 8 × NVIDIA A100-SXM4-80GB GPU
- 至少200GB磁盘空间

## 安装步骤

### 1. 克隆/下载项目

```bash
cd /Users/yuch3n/qwen3_nl2sql_grpo
```

### 2. 创建虚拟环境

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. 升级pip

```bash
pip install --upgrade pip setuptools wheel
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

**预期时间**: 5-10分钟（取决于网络）

### 5. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

**期望输出**:
```
PyTorch: 2.9.1
Transformers: 4.57.5
TRL: 0.26.2
CUDA可用: True
```

### 6. 配置W&B（可选但推荐）

```bash
pip install wandb
wandb login
# 输入你的API密钥
```

## 配置调整

### 修改数据路径

编辑 `config.yaml`:

```yaml
data:
  train_file: "/path/to/your/train.json"
  test_file: "/path/to/your/test.json"
```

### 修改GPU配置

编辑 `config.yaml`:

```yaml
system:
  num_gpus: 8
  gpu_ids: "0,1,2,3,4,5,6,7"  # 根据实际情况调整
```

或者设置环境变量:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 修改训练参数

编辑 `config.yaml`:

```yaml
training:
  num_train_epochs: 3           # 训练轮数
  per_device_train_batch_size: 8  # 批大小
  learning_rate: 7.3e-6         # 学习率
  max_steps: -1                 # -1表示不限制步数
```

## 验证设置

运行验证脚本：

```bash
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print('✓ PyTorch版本:', torch.__version__)
print('✓ CUDA可用:', torch.cuda.is_available())
print('✓ CUDA设备数:', torch.cuda.device_count())
print('✓ 设备名:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# 测试模型加载（需要时间）
# model_name = 'Qwen/Qwen2.5-14B-Instruct'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print('✓ 分词器加载成功')
"
```

## 快速开始

```bash
# 1. 准备数据
bash prepare_data.sh

# 2. 启动训练
bash run_training.sh

# 3. 评估模型（训练完成后）
bash run_evaluation.sh
```

## 常见安装问题

### 问题1：PyTorch CUDA版本不匹配
```
RuntimeError: CUDA runtime error (804) : unknown error
```

**解决**：
```bash
# 查看CUDA版本
nvcc --version

# 卸载PyTorch
pip uninstall torch

# 重新安装指定CUDA版本
pip install torch==2.9.1 torchvision==0.18.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

### 问题2：Hugging Face模型下载缓慢

设置缓存目录：

```bash
export HF_HOME=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache/datasets
```

或在脚本开头添加：

```python
import os
os.environ['HF_HOME'] = '/path/to/cache'
```

### 问题3：内存不足

减小批大小或增加梯度累积步数：

```yaml
training:
  per_device_train_batch_size: 4  # 从8改为4
  gradient_accumulation_steps: 4  # 从2改为4
```

有效批大小保持不变：4 × 4 = 16 per GPU

### 问题4：权限错误

```bash
chmod +x run_training.sh run_evaluation.sh prepare_data.sh
```

## 依赖包详解

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.9.1 | 深度学习框架 |
| transformers | 4.57.5 | Hugging Face模型 |
| trl | 0.26.2 | 强化学习训练（GRPO） |
| peft | 0.18.1 | 参数高效微调 |
| accelerate | 1.12.0 | 分布式训练 |
| wandb | 0.24.0 | 实验追踪 |
| tqdm | 4.66.2+ | 进度条 |
| pyyaml | 6.0+ | 配置文件解析 |

## 硬件规格

### 最低配置
- 1 × NVIDIA A100-80GB GPU
- 256GB 内存
- 200GB 磁盘

### 推荐配置（本项目）
- 8 × NVIDIA A100-SXM4-80GB GPU
- 512GB 内存
- 500GB 磁盘

### 网络
- 稳定的互联网连接（用于下载模型）

## 文件夹结构验证

安装后应该看到：

```
/Users/yuch3n/qwen3_nl2sql_grpo/
├── venv/                      # 虚拟环境
├── config.yaml                # ✓ 配置文件
├── requirements.txt           # ✓ 依赖
├── data/
│   ├── __init__.py
│   └── data_loader.py         # ✓ 数据模块
├── classifiers/
│   ├── __init__.py
│   ├── complexity_classifier.py
│   └── meta_classifier.py     # ✓ 分类器
├── generator/
│   ├── __init__.py
│   ├── prompts.py
│   └── sql_generator.py       # ✓ 生成器
├── reward/
│   ├── __init__.py
│   └── reward_model.py        # ✓ 奖励模型
├── training/
│   ├── __init__.py
│   ├── train_grpo.py
│   └── train_utils.py         # ✓ 训练脚本
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   └── metrics.py             # ✓ 评估模块
├── scripts/
│   └── prepare_data.py        # ✓ 数据准备
├── run_training.sh            # ✓ 训练脚本
├── run_evaluation.sh          # ✓ 评估脚本
├── prepare_data.sh            # ✓ 数据脚本
├── README.md                  # ✓ 文档
├── QUICK_START.md             # ✓ 快速开始
└── outputs/
    ├── checkpoints/           # （训练后生成）
    ├── logs/                  # （训练后生成）
    └── cache/                 # （自动生成）
```

## 下一步

1. ✅ 安装完成
2. ⏭️ 准备数据：`bash prepare_data.sh`
3. ⏭️ 开始训练：`bash run_training.sh`
4. ⏭️ 查看快速开始：`cat QUICK_START.md`

---

有问题？查看完整文档：`cat README.md`
