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

