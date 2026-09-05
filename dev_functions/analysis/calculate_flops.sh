#!/bin/bash

# 检查参数数量
if [ "$#" -ne 2 ]; then
    echo "使用方法: $0 <total_flos> <global_step>"
    echo "示例: $0 5.59169563686912e+18 4764"
    exit 1
fi

TOTAL_FLOS=$1
GLOBAL_STEP=$2

# 使用 Python 进行高精度计算并格式化输出
python -c "
total = float('$TOTAL_FLOS')
step = float('$GLOBAL_STEP')
flops = total / step

tflops = flops / 1e12
pflops = flops / 1e15

print(f'计算结果:')
print(f'  - 原始 FLOPS: {flops:.2f}')
print(f'  - 算力资源:   {tflops:.2f} TFLOPS')
print(f'  - 算力资源:   {pflops:.4f} PFLOPS')
"
