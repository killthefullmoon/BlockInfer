#!/bin/bash
#SBATCH --job-name=llada_benchmark
#SBATCH --account=zmao98                           # 替换为你的账户名
#SBATCH --partition=spgpu                          # 使用spgpu分区（GPU节点）
#SBATCH --gpus=2                                   # 请求2块GPU以启用多卡并行
#SBATCH --cpus-per-task=8                          # 每个任务使用8个CPU核心
#SBATCH --mem=64gb                                 # 请求64GB内存（LLaDA-8B需要约40-50GB）
#SBATCH --time=16:00:00                            # 设置作业最长运行16小时
#SBATCH --nodes=1                                  # 使用1个节点
#SBATCH --mail-type=START,END,FAIL                 # 作业开始、结束、失败时发送邮件
#SBATCH --output=logs/job_%j.out                   # 标准输出日志，%j将被替换为作业ID
#SBATCH --error=logs/job_%j.err                    # 错误信息日志

# Change to working directory
cd /nfs/turbo/coe-zmao/hymanzzs/BlockInfer

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /home/hymanzzs/miniforge3/etc/profile.d/conda.sh
conda activate llada

# Set environment
export HF_HOME=/nfs/turbo/coe-zmao/hymanzzs/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

# Run the shell script
./run_one_by_one.sh
