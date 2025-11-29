import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Optional[AutoConfig] = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mask_token_id: int = -1
    block_length: int = 4
    
    # Model type and strategy configuration
    model_type: str = "auto"  # "auto", "sdar", "llada", "dream"
    sampling_method: str = "auto"  # "auto", "standard", "gumbel"
    use_kv_cache: bool = True
    use_cuda_graphs: bool = True

    def __post_init__(self):
        if not os.path.isdir(self.model):
            try:
                # Allow passing a Hugging Face repo id by downloading a snapshot to the local cache
                from huggingface_hub import snapshot_download
                self.model = snapshot_download(repo_id=self.model, local_files_only=False)
            except Exception as exc:
                raise AssertionError(
                    f"Model path '{self.model}' does not exist and could not be downloaded."
                ) from exc
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        
        # Auto-detect model type if not specified
        if self.model_type == "auto":
            model_name_lower = self.model.lower()
            hf_model_type = getattr(self.hf_config, "model_type", "").lower()
            
            if "llada" in model_name_lower or "llada" in hf_model_type:
                self.model_type = "llada"
            elif "dream" in model_name_lower or "dream" in hf_model_type:
                self.model_type = "dream"
            elif "sdar" in hf_model_type:
                self.model_type = "sdar"
            else:
                # Default to sdar for backward compatibility
                self.model_type = "sdar"
        
        # Configure sampling method based on model type
        if self.sampling_method == "auto":
            if self.model_type == "llada":
                self.sampling_method = "gumbel"
            else:
                self.sampling_method = "standard"
        
        # Configure KV cache and CUDA graphs based on model type
        if self.model_type in ("llada", "dream"):
            # LLaDA/Dream don't use KV cache or CUDA graphs in vanilla implementations
            self.use_kv_cache = False
            self.use_cuda_graphs = False
            # Set a minimal value for BlockManager (not actually used for KV cache)
            if self.num_kvcache_blocks == -1:
                self.num_kvcache_blocks = 100
        
        # Some configs (e.g., LLaDA) expose max_sequence_length instead of max_position_embeddings
        max_pos = getattr(self.hf_config, "max_position_embeddings", None)
        if max_pos is None:
            max_pos = getattr(self.hf_config, "max_sequence_length", self.max_model_len)
        self.max_model_len = min(self.max_model_len, max_pos)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.mask_token_id != -1, "Mask token ID must be set"
