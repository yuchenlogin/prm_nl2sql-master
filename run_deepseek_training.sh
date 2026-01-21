#!/bin/bash
# DeepSeek GRPO 训练脚本
# 使用方法: bash run_deepseek_training.sh

set -e

echo "=========================================="
echo "NL2SQL DeepSeek GRPO 训练脚本"
echo "=========================================="

# 智能GPU调度器
GPU_SCHEDULER="/lpai/prm_nl2sql/prm_venv/bin/python3 utils/gpu_scheduler.py"
MEMORY_THRESHOLD=30  # 显存使用率阈值
MIN_GPUS=1          # 最少需要的GPU
MAX_GPUS=8          # 最多使用的GPU

echo ""
echo "=== 智能GPU调度 ==="
echo "使用GPU调度器选择可用的GPU..."

# 运行GPU调度器获取配置
CONFIG_OUTPUT=$($GPU_SCHEDULER --memory-threshold $MEMORY_THRESHOLD --min-gpus $MIN_GPUS --max-gpus $MAX_GPUS --get-config 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$CONFIG_OUTPUT" ]; then
    echo "GPU配置加载成功:"
    echo "$CONFIG_OUTPUT"

    # 提取配置变量
    CUDA_VISIBLE_DEVICES=$(echo "$CONFIG_OUTPUT" | grep 'CUDA_VISIBLE_DEVICES' | cut -d'=' -f2)
    N_GPUS_PER_NODE=$(echo "$CONFIG_OUTPUT" | grep 'N_GPUS_PER_NODE' | cut -d'=' -f2)
    TENSOR_MP_SIZE=$(echo "$CONFIG_OUTPUT" | grep 'TENSOR_MODEL_PARALLEL_SIZE' | cut -d'=' -f2)

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
else
    echo "警告: GPU调度器失败，使用默认配置"
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    N_GPUS_PER_NODE=4
    TENSOR_MP_SIZE=2
fi

echo ""
echo "最终GPU配置:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  GPU Count: ${N_GPUS_PER_NODE}"
echo "  TENSOR_MODEL_PARALLEL_SIZE: ${TENSOR_MP_SIZE}"
echo ""

# 创建输出目录
mkdir -p outputs/logs outputs/deepseek_checkpoints outputs/deepseek_proof_pool

# 训练日志文件
LOG_FILE="outputs/logs/deepseek_training_$(date +%Y%m%d_%H%M%S).log"

echo "启动时间: $(date)"
echo "训练日志: $LOG_FILE"
echo ""

# 启动训练 - 传递GPU配置到训练脚本，并重定向日志
PYTHONPATH=/lpai/prm_nl2sql:$PYTHONPATH /lpai/prm_nl2sql/prm_venv/bin/python3 training/train_deepseek_grpo.py --config config_deepseek.yaml \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --gpus_per_node "$N_GPUS_PER_NODE" \
    --tensor_model_parallel_size "$TENSOR_MP_SIZE" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "训练完成时间: $(date)"
echo "=========================================="
echo "最佳模型位置: ./outputs/deepseek_checkpoints/final_model/"
echo "训练日志位置: ./outputs/logs/deepseek_training_*.log"
echo ""
echo "下一步: 运行评估"
echo "bash run_evaluation.sh"
