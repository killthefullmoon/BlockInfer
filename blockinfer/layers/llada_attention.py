"""
Standard bidirectional attention for LLaDA models.
Uses flash attention without the staircase masking used in BlockAttention.
"""

import torch
from torch import nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from blockinfer.utils.context import get_context
from blockinfer.engine.sequence import RunType


class LLaDABidirectionalAttention(nn.Module):
    """
    Bidirectional attention for LLaDA using standard flash attention.
    Unlike BlockAttention (which uses sparse/staircase patterns for SDAR),
    this implements true full bidirectional attention for masked diffusion.
    """
    
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Forward pass with full bidirectional attention.
        
        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            k: Key tensor [total_tokens, num_kv_heads, head_dim]
            v: Value tensor [total_tokens, num_kv_heads, head_dim]
        """
        # Reshape to expected format
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        if context.run_type == RunType.PREFILL:
            # PREFILL: Use full bidirectional attention on the entire sequence
            # No causal masking, no staircase - true bidirectional
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=False,  # CRITICAL: Bidirectional, not causal
            )
            
            # Store to KV cache if available
            if k_cache.numel() and v_cache.numel() and hasattr(context, 'slot_mapping'):
                from blockinfer.layers.attention import store_kvcache
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        else:  # DENOISE
            # DENOISE: Query the current block against all cached context
            q = q.view(-1, context.block_length, self.num_heads, self.head_dim)
            k = k.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            v = v.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            
            o = flash_attn_with_kvcache(
                q,
                k_cache=k_cache,
                v_cache=v_cache,
                k=k,
                v=v,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=False,  # Bidirectional
            )
        
        # Reshape output
        o = o.view(-1, self.num_heads * self.head_dim)
        return o

