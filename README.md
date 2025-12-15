# BlockInfer: Block-wise Diffusion for LLaDA

BlockInfer is an efficient inference framework for **LLaDA** (Large Language Diffusion with mAsking) that implements block-wise diffusion decoding with paged attention and KV-cache optimization.

## Overview

This benchmark compares two inference strategies for LLaDA:

| Method | Description | Block Size |
|--------|-------------|------------|
| **BlockInfer** | Block-wise diffusion with paged attention | `block_length` (e.g., 32) |
| **Vanilla LLaDA** | Standard diffusion (all tokens at once) | `max_tokens` |

## Requirements

```bash
# Create conda environment
conda create -n blockinfer python=3.10
conda activate blockinfer

# Install PyTorch (CUDA 11.8+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers accelerate triton
```

## Quick Start

### 1. Run Benchmark (Local)

```bash
# Basic benchmark with 50 samples
python benchmark_llada.py --num-samples 50

# Specify max tokens and block length
python benchmark_llada.py --max-tokens 128 --block-length 32 --num-samples 100

# Filter by dataset source
python benchmark_llada.py --source gsm8k --num-samples 50

# Skip vanilla LLaDA (only run BlockInfer)
python benchmark_llada.py --skip-vanilla --num-samples 50
```

### 2. Run on SLURM Cluster

```bash
# Submit job
sbatch run_benchmark.slurm

# Check job status
squeue -u $USER

# View logs
tail -f logs/job_*.out
```

## Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `sample_200.jsonl` | Dataset file path |
| `--num-samples` | All | Number of samples to benchmark |
| `--source` | None | Filter by source: `gsm8k`, `math500`, `mmlu`, `longbench_hotpotqa` |
| `--max-tokens` | 128 | Maximum tokens to generate |
| `--block-length` | 32 | Block length for BlockInfer |
| `--denoising-steps` | 32 | Denoising steps per block |
| `--skip-vanilla` | False | Skip vanilla LLaDA benchmark |
| `--skip-sample` | 0 | Skip first N samples |
| `--output` | `benchmark_results.json` | Output file |

## Example Commands

```bash
# Full benchmark on GSM8K
python benchmark_llada.py \
    --source gsm8k \
    --max-tokens 256 \
    --block-length 32 \
    --num-samples 100 \
    --output benchmark_gsm8k.json

# Quick test (10 samples, BlockInfer only)
python benchmark_llada.py \
    --num-samples 10 \
    --skip-vanilla \
    --max-tokens 128

# Benchmark with different block sizes
for block in 16 32 64; do
    python benchmark_llada.py \
        --block-length $block \
        --num-samples 50 \
        --skip-vanilla \
        --output benchmark_block_${block}.json
done
```


## Dataset

The benchmark uses `sample_200.jsonl` containing questions from:

- **GSM8K**: Math word problems
- **MATH500**: Advanced math
- **MMLU**: Multiple choice QA
- **LongBench-HotpotQA**: Long-context QA

## Project Structure

```
BlockInfer/
├── benchmark_llada.py      # Main benchmark script
├── blockinfer/             # Core inference engine
│   ├── llm.py              # LLM interface
│   ├── engine/             # Scheduler, block manager
│   ├── kernels/            # Triton attention kernels
│   ├── layers/             # Model layers
│   └── models/             # LLaDA model implementations
├── sample_200.jsonl        # Benchmark dataset
├── run_benchmark.slurm     # SLURM job script
└── utils.py                # Utility functions
```
