"""
Vanilla LLaDA model wrapper for BlockInfer.
This implementation directly uses the transformers LLaDA model without custom layers.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoConfig


class LLaDAVanillaForCausalLM(nn.Module):
    """
    Vanilla wrapper for LLaDA that uses the original transformers implementation.
    This avoids custom layers that haven't been thoroughly tested.
    """
    
    def __init__(self, config_or_path: str):
        super().__init__()
        
        # Load the vanilla LLaDA model from transformers
        if isinstance(config_or_path, str):
            # Load from pretrained
            self.model = AutoModelForCausalLM.from_pretrained(
                config_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            self.config = self.model.config
        else:
            # Initialize from config
            self.config = config_or_path
            self.model = AutoModelForCausalLM.from_config(
                config_or_path,
                trust_remote_code=True
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        **kwargs
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (seq_len,) or (batch_size, seq_len)
            position_ids: Position IDs (ignored for LLaDA, kept for compatibility)
            past_key_values: Cached key-value pairs for faster generation
            use_cache: Whether to return key-value cache
            
        Returns:
            Model outputs with logits
        """
        # LLaDA model expects input_ids with shape (batch_size, seq_len)
        # If 1D, add batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # LLaDA doesn't use position_ids explicitly - RoPE is computed internally
        outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        
        # Return the full outputs object
        return outputs
    
    def compute_logits(self, outputs) -> torch.Tensor:
        """
        Extract logits from model outputs.
        
        Args:
            outputs: Model outputs object from forward()
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # LLaDA outputs have a .logits attribute
        if hasattr(outputs, 'logits'):
            return outputs.logits
        else:
            raise AttributeError("Cannot find logits in model outputs")

