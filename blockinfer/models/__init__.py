"""
BlockInfer model implementations.

This module contains model implementations for various architectures:
- SDAR: Speculative Decoding with Autoregressive models
- SDARMoE: SDAR with Mixture of Experts
- LLaDA: Large Language Diffusion with Masking
- Dream: Dream diffusion model
"""

from blockinfer.models.sdar import SDARForCausalLM
from blockinfer.models.sdar_moe import SDARMoeForCausalLM
from blockinfer.models.llada_vanilla import LLaDAVanillaForCausalLM
from blockinfer.models.dream_vanilla import DreamVanillaForCausalLM

__all__ = [
    "SDARForCausalLM",
    "SDARMoeForCausalLM", 
    "LLaDAVanillaForCausalLM",
    "DreamVanillaForCausalLM",
]
