#!/bin/bash
# 评估脚本
# 使用方法: bash run_evaluation.sh

set -e

echo "=========================================="
echo "NL2SQL 模型评估脚本"
echo "=========================================="

# 智能GPU调度器
GPU_SCHEDULER="python3 utils/gpu_scheduler.py"
MEMORY_THRESHOLD=30  # 显存使用率阈值
MIN_GPUS=1          # 最少需要的GPU
MAX_GPUS=4          # 最多使用的GPU（评估通常不需要太多GPU）

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
    export CUDA_VISIBLE_DEVICES="0"
    N_GPUS_PER_NODE=1
    TENSOR_MP_SIZE=1
fi

echo ""
echo "最终GPU配置:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  GPU Count: ${N_GPUS_PER_NODE}"
echo ""

# 默认路径
MODEL_PATH="./outputs/checkpoints/best_model"
TEST_FILE="./data/nl2_sql_cold_start_sft_all_test_swift_830_1231.json"
OUTPUT_DIR="./outputs/evaluation"

# 支持命令行参数覆盖
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型不存在 ($MODEL_PATH)"
    echo "请先运行训练: bash run_training.sh"
    echo "或者指定模型路径: bash run_evaluation.sh --model /path/to/model"
    exit 1
fi

# 检查测试文件是否存在
if [ ! -f "$TEST_FILE" ]; then
    echo "错误: 测试文件不存在 ($TEST_FILE)"
    exit 1
fi

echo "============================================================"
echo "评估配置"
echo "============================================================"
echo "模型路径: $MODEL_PATH"
echo "测试文件: $TEST_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "GPU配置: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "启动时间: $(date)"
echo "============================================================"
echo ""

# 运行评估
python3 evaluation/evaluator.py \
    --model "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output "$OUTPUT_DIR/evaluation_report.json" \
    --batch_size 16 \
    --max_length 1024

EVAL_STATUS=$?

echo ""
echo "=========================================="
if [ $EVAL_STATUS -eq 0 ]; then
    echo "✅ 评估完成时间: $(date)"
else
    echo "❌ 评估失败，状态码: $EVAL_STATUS"
fi
echo "=========================================="
echo "评估报告位置: $OUTPUT_DIR/evaluation_report.json"
echo ""

# 显示评估结果摘要
if [ -f "$OUTPUT_DIR/evaluation_report.json" ]; then
    echo "=========================================="
    echo "评估结果摘要"
    echo "=========================================="
    echo ""
    cat "$OUTPUT_DIR/evaluation_report.json" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    summary = data.get('summary', {})
    for key, value in summary.items():
        print(f'{key}: {value}')
except:
    print('无法解析评估结果')
"
    echo ""
    echo "完整结果查看:"
    echo "cat $OUTPUT_DIR/evaluation_report.json"
else
    echo "⚠️ 警告: 评估报告未生成"
fi

echo "=========================================="
