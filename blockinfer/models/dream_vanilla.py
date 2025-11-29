"""
Vanilla Dream model wrapper for BlockInfer.
Uses the HF implementation directly.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class DreamVanillaForCausalLM(nn.Module):
    """
    Simple wrapper around the Dream diffusion model provided by HF.
    For BlockInfer we use the same interface as LLaDA: forward takes only input_ids.
    """

    def __init__(self, config_or_path: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            config_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.config = self.model.config

    def forward(self, input_ids: torch.Tensor, **kwargs):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.model(input_ids=input_ids, use_cache=False, **kwargs)

    def diffusion_generate(self, *args, **kwargs):
        """
        Proxy to the underlying Dream model's diffusion_generate.
        """
        if hasattr(self.model, "diffusion_generate"):
            return self.model.diffusion_generate(*args, **kwargs)
        raise AttributeError("Underlying Dream model does not provide diffusion_generate")

    def compute_logits(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "logits"):

            return outputs.logits
        raise AttributeError("Cannot find logits in model outputs")
