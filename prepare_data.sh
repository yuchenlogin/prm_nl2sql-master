#!/bin/bash
# 数据准备脚本
# 使用方法: bash prepare_data.sh

set -e

echo "=========================================="
echo "NL2SQL 数据准备脚本"
echo "=========================================="

# 数据文件路径
TRAIN_FILE="./data/nl2_sql_cold_start_sft_all_train_swift_9501_1231.json"
TEST_FILE="./data/nl2_sql_cold_start_sft_all_test_swift_830_1231.json"
OUTPUT_DIR="./outputs"

# 检查数据文件是否存在
if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练文件不存在 ($TRAIN_FILE)"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "错误: 测试文件不存在 ($TEST_FILE)"
    exit 1
fi

echo "训练文件: $TRAIN_FILE"
echo "测试文件: $TEST_FILE"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行数据准备
python scripts/prepare_data.py \
    --train_file "$TRAIN_FILE" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "数据准备完成"
echo "=========================================="
echo "报告位置: $OUTPUT_DIR/data_preparation_report.json"
