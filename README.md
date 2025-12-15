# BlockInfer

## Installation
### Environment Setup

```
transformers>=4.52.4
flash-attn
```

For Local Inference:

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Labman42/BlockInfer.git
cd BlockInfer
pip install .
```
For RL training usage (support DP and TP, managed by accelerate from huggingface):

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Labman42/BlockInfer.git
cd BlockInfer
git checkout accelerate
pip install .
```

## Quick Start

```bash
python example.py
```

See `example.py` for usage. The API mirrors vLLM's interface with some differences in the `LLM.generate` method.

## Benchmarks

### Dream diffusion benchmark
Run `dreams_benchmark.py` over a JSONL dataset containing `question`, `answer`, and `source` fields.

```bash
python dreams_benchmark.py \
  --dataset sample_200.jsonl \
  --source math500 \
  --max-samples 50 \
  --max-new-tokens 256 \
  --steps 32
```

- Default model is `Dream-org/Dream-v0-Instruct-7B` (uses local HF cache if available).
- Supported sources include `gsm8k`, `math500`, `mmlu`, `longbench_hotpotqa`; `--source` filters which rows are evaluated.

### SDAR benchmark (BlockInfer vs vanilla)
Compare BlockInfer SDAR with the vanilla `diffusion_generate` using `SDAR_benchmark.py`.

```bash
python SDAR_benchmark.py \
  --model-path /path/to/SDAR-4B-Chat \
  --dataset sample_200.jsonl \
  --sources gsm8k \
  --max-tokens 256 \
  --block-length 4 \
  --denoising-steps 4
```

- Dataset rows should include `question`, `answer`, `source`, and optional `choices` (for `mmlu`).
- Use `--dataset-name` to target a single source; `--use-streaming` and `--max-active` enable streaming for BlockInfer.
- `--skip-blockinfer` or `--skip-vanilla` skip either side of the comparison; results are written to `benchmark_sdar_results.json` (or `--output`).
