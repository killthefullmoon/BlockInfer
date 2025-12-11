"""
Vanilla Dream model wrapper for BlockInfer.
This implementation directly uses the transformers Dream model without custom layers.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoConfig


class DreamVanillaForCausalLM(nn.Module):
    """
    Vanilla wrapper for Dream that uses the original transformers implementation.
    This avoids custom layers that haven't been thoroughly tested.
    """
    
    def __init__(self, config_or_path: str):
        super().__init__()
        
        # Load the vanilla Dream model from transformers
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        tok_idx: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (seq_len,) or (batch_size, seq_len)
            attention_mask: Attention mask for padding
            position_ids: Position IDs (ignored for Dream, kept for compatibility)
            past_key_values: Cached key-value pairs for faster generation
            use_cache: Whether to return key-value cache
            tok_idx: Token indices for Dream model
            
        Returns:
            Model outputs with logits
        """
        # Dream model expects input_ids with shape (batch_size, seq_len)
        # If 1D, add batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Build kwargs for Dream model
        model_kwargs = {
            'past_key_values': past_key_values,
            'use_cache': use_cache,
        }
        
        # Add attention_mask if provided
        if attention_mask is not None:
            model_kwargs['attention_mask'] = attention_mask
        
        # Add tok_idx if provided (Dream-specific)
        if tok_idx is not None:
            model_kwargs['tok_idx'] = tok_idx
        
        # Add any additional kwargs
        model_kwargs.update(kwargs)
        
        # Run model forward
        outputs = self.model(input_ids, **model_kwargs)
        
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
        # Dream outputs have a .logits attribute
        if hasattr(outputs, 'logits'):
            return outputs.logits
        else:
            raise AttributeError("Cannot find logits in model outputs")
