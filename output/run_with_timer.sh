#!/bin/bash

# 获取脚本参数
PYTHON_CMD="$1"
LOG_FILE="$2"

# 记录开始时间
START_TIME=$(date +%s)
echo "开始训练时间: $(date)" | tee -a "$LOG_FILE"

# 执行训练命令
eval "$PYTHON_CMD" 2>&1 | tee -a "$LOG_FILE"

# 记录结束时间
END_TIME=$(date +%s)
echo "结束训练时间: $(date)" | tee -a "$LOG_FILE"

# 计算总运行时间
TOTAL_SECONDS=$((END_TIME - START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=$((TOTAL_SECONDS % 60))

# 输出总运行时间
echo "====================================" | tee -a "$LOG_FILE"
echo "总训练时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒" | tee -a "$LOG_FILE"
echo "====================================" | tee -a "$LOG_FILE"
