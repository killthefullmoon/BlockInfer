#!/bin/bash
#SBATCH --job-name=LLaDA_BlockInfer_Benchmark      # 作业名称
#SBATCH --account=zmao98                           # 替换为你的账户名
#SBATCH --partition=spgpu                          # 使用spgpu分区（GPU节点）
#SBATCH --gpus=1                                   # 请求1块GPU（LLaDA需要1块GPU）
#SBATCH --cpus-per-task=8                          # 每个任务使用8个CPU核心
#SBATCH --mem=64gb                                 # 请求64GB内存（LLaDA-8B需要约40-50GB）
#SBATCH --time=16:00:00                            # 设置作业最长运行16小时
#SBATCH --nodes=1                                  # 使用1个节点
#SBATCH --mail-type=START,END,FAIL                 # 作业开始、结束、失败时发送邮件
#SBATCH --output=logs/job_%j.out                   # 标准输出日志，%j将被替换为作业ID
#SBATCH --error=logs/job_%j.err                    # 错误信息日志

source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate dllm-rl  # 替换为你的环境名

python benchmark_llada.py --max-tokens 256
