"""
Utility functions for BlockInfer testing and usage.
"""

import os
from pathlib import Path


def find_cached_model(model_id: str) -> str | None:
    """
    Return the latest local HF cache snapshot path for the given model_id, if it exists.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "GSAI-ML/LLaDA-8B-Instruct")
    
    Returns:
        Path to the cached model if found, None otherwise
    """
    cache_dir = None
    try:
        from huggingface_hub import file_download
        cache_dir = getattr(file_download, "HUGGINGFACE_HUB_CACHE", None)
    except Exception:
        cache_dir = None
    
    # Fallback to HF_HOME if set, or default cache location
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = os.path.join(hf_home, "hub")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    base = Path(cache_dir)
    snap_root = base / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snap_root.is_dir():
        return None
    
    # Pick the most recent snapshot that has a config.json
    snapshots = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in snapshots:
        if (snap / "config.json").is_file():
            return str(snap)
    return None

