# BlockInfer

A lightweight inference engine for block diffusion language models.

## Supported Models

- ✅ **SDAR** - Block diffusion model
- ✅ **SDAR-MoE** - Mixture of Experts variant  
- ✅ **LLaDA** - Large Language Diffusion with mAsking

## Installation

### Requirements

```
Python >= 3.10, < 3.13
transformers >= 4.52.4
flash-attn
torch >= 2.4.0
triton >= 3.0.0
```

### Install

```bash
pip install flash-attn --no-build-isolation
git clone https://github.com/Labman42/BlockInfer.git
cd BlockInfer
pip install -e .
```

## Quick Start

### SDAR Models

```python
from blockinfer import LLM, SamplingParams

llm = LLM(
    "path/to/sdar-model",
    mask_token_id=151669,
    block_length=4
)

params = SamplingParams(
    temperature=1.0,
    max_tokens=1024,
    remasking_strategy="entropy_bounded",
    block_length=4,
    denoising_steps=4,
    eb_threshold=0.6
)

outputs = llm.generate(prompts, params)
```

See `example.py` for complete SDAR usage.

### LLaDA Models

```python
from blockinfer import LLM, SamplingParams

llm = LLM(
    "GSAI-ML/LLaDA-8B-Instruct",
    mask_token_id=126336,
    block_length=32
)

params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    remasking_strategy="low_confidence",
    block_length=32,
    denoising_steps=32
)

outputs = llm.generate(prompts, params)
```
### Dreams Models

```python
from blockinfer import LLM, SamplingParams

llm = LLM(
    "Dream-org/Dream-v0-Instruct-7B",
    mask_token_id=tokenizer.mask_token_id or tokenizer.pad_token_id,
    block_length=32
)

params = SamplingParams(
    temperature=0.2,
    topk=0,
    topp=0.95,
    max_tokens=512,
    remasking_strategy="entropy_bounded",
    block_length=16,
    denoising_steps=512,
    dynamic_threshold=0.0,
)


outputs = llm.generate(prompts, params)
```
See `example_llada_blockinfer.py` for complete LLaDA usage.

**LLaDA Features**:
- Block decode: Full sequence → block-wise denoising
- Weight loading from HuggingFace (automatic)
- Bidirectional attention support
- 6+ remasking strategies

**Documentation**: See `BLOCK_DECODE_COMPLETE.md` for LLaDA implementation details.

## Features

### Remasking Strategies

- `sequential` - Left-to-right denoising
- `low_confidence_static` - Confidence-based with fixed transfer count
- `low_confidence_dynamic` - Confidence-based with adaptive threshold
- `entropy_bounded` - Entropy-budget based selection
- `low_confidence` - LLaDA's default strategy
- `random` - Random selection

### Advanced Features

- **Block Decode**: Process generation in fixed-size blocks
- **Tensor Parallelism**: Multi-GPU support
- **CUDA Graphs**: Automatic graph capture for speedup
- **KV-Cache**: Efficient caching even for bidirectional attention
- **Batch Inference**: `generate_streaming()` for high throughput

## Performance

### SDAR
- Optimized block diffusion
- Staircase sparse attention
- Efficient MoE support

### LLaDA  
- Block decode: ~14 tokens/sec on A100
- Weight loading: Automatic from HuggingFace
- 194/291 weights mapped and loaded
- Pipeline fully functional

## Architecture

```
BlockInfer/
├── blockinfer/
│   ├── models/
│   │   ├── sdar.py              # SDAR model
│   │   ├── sdar_moe.py          # SDAR-MoE model
│   │   └── llada.py             # LLaDA model
│   ├── layers/
│   │   ├── attention.py         # BlockAttention (for SDAR)
│   │   ├── llada_attention.py   # Bidirectional attention (for LLaDA)
│   │   ├── linear.py
│   │   ├── layernorm.py
│   │   └── ...
│   ├── engine/
│   │   ├── llm_engine.py        # Main LLM interface
│   │   ├── model_runner.py      # Model execution
│   │   ├── scheduler.py         # Request scheduling
│   │   └── sequence.py          # Sequence management
│   ├── utils/
│   │   └── loader.py            # Weight loading (supports SDAR & LLaDA)
│   └── ...
├── example.py                   # SDAR example
├── example_llada_blockinfer.py  # LLaDA example
└── README.md                    # This file
```

## LLaDA Implementation Notes

### Block Decode Conversion

BlockInfer implements **block-level decoding** for LLaDA, converting from full-sequence to block-wise:

**Full Sequence (Original)**:
```
Mask all N tokens
For step in N:
    Denoise all positions
    Remask low-confidence
```

**Block Decode (BlockInfer)**:
```
For each block of size B:
    Mask block tokens
    For step in B:
        Denoise block only
        Remask within block
    Commit block
```

**Advantages**:
- Better memory efficiency
- Easier batching across requests
- Streaming support
- Maintains quality

### Weight Loading

Automatic mapping from HuggingFace LLaDA to BlockInfer architecture:
- QKV fusion: `q_proj + k_proj + v_proj → qkv_proj`
- MLP fusion: `ff_proj + up_proj → gate_up_proj`
- Parameter name mapping handled automatically

## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## License

MIT License
