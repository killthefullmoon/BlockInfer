import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
import torch.distributed as dist


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def _register_empty_parameter(module, name: str):
    empty = nn.Parameter(torch.empty(0, device='meta'), requires_grad=False)
    module.register_parameter(name, empty)

def _prepare_fused_tensors(model: nn.Module, device: torch.device | str = "cuda"):
    for module in model.modules():
        if not (hasattr(module, "experts") and module.experts):
            continue

        exp0 = module.experts[0]
        if hasattr(exp0, "gate_up_proj"):
            # gate_up_proj weight shape (2 * intermediate_local, hidden_size)
            shape = (len(module.experts),) + exp0.gate_up_proj.weight.shape
            module.register_buffer("_w1",
                                   torch.empty(shape,
                                               dtype=exp0.gate_up_proj.weight.dtype,
                                               device=device),
                                   persistent=False)

        if hasattr(exp0, "down_proj"):
            # down_proj weight shape (hidden_size, intermediate_local)
            shape = (len(module.experts),) + exp0.down_proj.weight.shape
            module.register_buffer("_w2",
                                   torch.empty(shape,
                                               dtype=exp0.down_proj.weight.dtype,
                                               device=device),
                                   persistent=False)

        # Strip per-expert parameters to save VRAM (weights now live in fused buffers)
        for expert in module.experts:
            if hasattr(expert, "gate_up_proj"):
                _register_empty_parameter(expert.gate_up_proj, "weight")
            if hasattr(expert, "down_proj"):
                _register_empty_parameter(expert.down_proj, "weight")

def _is_moe_expert_weight(weight_name: str) -> bool:
    """Check if weight belongs to an MoE expert."""
    return 'experts.' in weight_name and ('gate_up_proj' in weight_name or 'down_proj' in weight_name)

def _load_expert_weight_to_fused(model: nn.Module, weight_name: str, weight_tensor: torch.Tensor, shard_id=None):
    """Load expert weight directly into the appropriate fused tensor with tensor parallel support.

    Tensor Parallel Rules:
      gate_up_proj (ColumnParallel): shard on dim0 (output dimension)
      gate_proj/up_proj shards (when merged loading) each cover half of dim0; we fill into fused _w1 slices
      down_proj (RowParallel): shard on dim1 (input dimension)
    """
    parts = weight_name.split('.')
    layer_path = []
    expert_idx = None
    proj_type = None

    for i, part in enumerate(parts):
        if part == 'experts':
            expert_idx = int(parts[i + 1])
            proj_type = parts[i + 2]  # gate_up_proj or down_proj
            layer_path = parts[:i]
            break

    if expert_idx is None:
        return

    # Resolve module
    moe_module = model
    for attr in layer_path:
        moe_module = getattr(moe_module, attr)

    tp_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    tp_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    if proj_type == 'gate_up_proj' and hasattr(moe_module, '_w1'):
        fused_tensor = moe_module._w1  # (E, 2*I_local, H)
        local_out = fused_tensor.shape[1]
        # Two cases:
        # 1. Loading merged tensor (gate+up) => shard_id is None, weight_tensor shape (2*I_global, H)
        # 2. Loading individual shard (gate or up) => shard_id in {0,1}, weight_tensor shape (I_global, H)
        if shard_id is None:
            # merged load
            if weight_tensor.shape[0] == local_out:
                # Already sharded
                fused_tensor[expert_idx].copy_(weight_tensor)
            else:
                assert weight_tensor.shape[0] % tp_size == 0 and (weight_tensor.shape[0] // tp_size) == local_out, \
                    f"Unexpected gate_up_proj merged shape {weight_tensor.shape} vs local {fused_tensor.shape} tp={tp_size}"
                local_weight = weight_tensor.narrow(0, tp_rank * local_out, local_out)
                fused_tensor[expert_idx].copy_(local_weight)
        else:
            # individual gate or up
            half_local = local_out // 2
            if weight_tensor.shape[0] == half_local:
                # Already sharded
                start_idx = shard_id * half_local
                fused_tensor[expert_idx, start_idx:start_idx + half_local].copy_(weight_tensor)
            else:
                global_half = weight_tensor.shape[0]
                assert global_half % tp_size == 0 and global_half // tp_size == half_local, \
                    f"Unexpected gate/up proj shard shape {weight_tensor.shape} expected per-rank {half_local}"
                local_weight = weight_tensor.narrow(0, tp_rank * half_local, half_local)
                start_idx = shard_id * half_local
                fused_tensor[expert_idx, start_idx:start_idx + half_local].copy_(local_weight)
    elif proj_type == 'down_proj' and hasattr(moe_module, '_w2'):
        fused_tensor = moe_module._w2  # (E, H, I_local)
        local_in = fused_tensor.shape[2]
        if weight_tensor.shape[1] == local_in:
            fused_tensor[expert_idx].copy_(weight_tensor)
        else:
            assert weight_tensor.shape[1] % tp_size == 0 and (weight_tensor.shape[1] // tp_size) == local_in, \
                f"Unexpected down_proj shape {weight_tensor.shape} vs local {fused_tensor.shape} tp={tp_size}"
            local_weight = weight_tensor.narrow(1, tp_rank * local_in, local_in)
            fused_tensor[expert_idx].copy_(local_weight)

def _create_llada_weight_mapping(num_layers: int):
    """Create weight mapping from HF LLaDA to BlockInfer LLaDA."""
    mapping = {
        # Global weights
        'model.transformer.wte.weight': 'model.embed_tokens.weight',
        'model.transformer.norm.weight': 'model.norm.weight',
        'model.transformer.ff_out.weight': 'lm_head.weight',
    }
    
    for i in range(num_layers):
        # Attention output (attn_out not attention.out_proj!)
        mapping[f'model.transformer.blocks.{i}.attn_out.weight'] = \
            f'model.layers.{i}.self_attn.o_proj.weight'
        
        # MLP down projection
        mapping[f'model.transformer.blocks.{i}.ff_out.weight'] = \
            f'model.layers.{i}.mlp.down_proj.weight'
        
        # Norms (attn_norm and ff_norm, not attention_norm!)
        mapping[f'model.transformer.blocks.{i}.attn_norm.weight'] = \
            f'model.layers.{i}.input_layernorm.weight'
        mapping[f'model.transformer.blocks.{i}.ff_norm.weight'] = \
            f'model.layers.{i}.post_attention_layernorm.weight'
        
        # Note: q_proj, k_proj, v_proj → qkv_proj (fused)
        # Note: ff_proj + up_proj → gate_up_proj (fused)
        # Handled separately in load logic
    
    return mapping


def _is_llada_model(model_path: str) -> bool:
    """Check if the model path points to a LLaDA model."""
    # Try local path first
    files = glob(os.path.join(model_path, "*.safetensors"))
    
    if not files:
        # Try HuggingFace model ID
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            files_in_repo = list_repo_files(model_path)
            safetensor_files = [f for f in files_in_repo if f.endswith('.safetensors')]
            if safetensor_files:
                # Download first file to check
                first_file = hf_hub_download(model_path, safetensor_files[0])
                files = [first_file]
        except:
            return False
    
    if not files:
        return False
    
    # Check if weights use LLaDA naming (model.transformer.blocks)
    try:
        with safe_open(files[0], framework="pt", device="cpu") as f:
            keys = list(f.keys())
            return any('model.transformer.blocks' in k or 'transformer.blocks' in k for k in keys[:20])
    except:
        return False


def load_model(model: nn.Module, path: str):
    _prepare_fused_tensors(model)

    # Check if this is a LLaDA model
    is_llada = _is_llada_model(path)
    
    if is_llada:
        print("Detected LLaDA model, using custom weight mapping...")
        num_layers = len([m for m in model.modules() if hasattr(m, 'self_attn')])
        llada_mapping = _create_llada_weight_mapping(num_layers)
        
        # Get safetensors files (local or download from HF)
        files = glob(os.path.join(path, "*.safetensors"))
        if not files:
            # Download from HuggingFace
            from huggingface_hub import snapshot_download
            print(f"Downloading model from HuggingFace: {path}")
            local_path = snapshot_download(path, allow_patterns="*.safetensors")
            files = glob(os.path.join(local_path, "*.safetensors"))
        
        # Load with LLaDA mapping
        loaded_count = 0
        
        # First pass: collect all weights into memory
        print("Loading all weights into memory...")
        weight_cache = {}
        for file in files:
            print(f"  Reading {os.path.basename(file)}...")
            with safe_open(file, "pt", "cpu") as f:
                for key in f.keys():
                    weight_cache[key] = f.get_tensor(key)
        
        print(f"✓ Loaded {len(weight_cache)} tensors into cache\n")
        
        # Second pass: map and copy weights
        print("Mapping weights to BlockInfer architecture...")
        
        # Direct mappings
        for hf_name, target_name in llada_mapping.items():
            if hf_name in weight_cache:
                try:
                    param = model.get_parameter(target_name)
                    param.data.copy_(weight_cache[hf_name])
                    loaded_count += 1
                except AttributeError:
                    pass
        
        # Fuse QKV weights
        qkv_count = 0
        for i in range(num_layers):
            # Correct naming: q_proj not attention.q_proj!
            q_key = f'model.transformer.blocks.{i}.q_proj.weight'
            k_key = f'model.transformer.blocks.{i}.k_proj.weight'
            v_key = f'model.transformer.blocks.{i}.v_proj.weight'
            
            if q_key in weight_cache and k_key in weight_cache and v_key in weight_cache:
                qkv_weight = torch.cat([
                    weight_cache[q_key],
                    weight_cache[k_key],
                    weight_cache[v_key]
                ], dim=0)
                
                target_name = f'model.layers.{i}.self_attn.qkv_proj.weight'
                try:
                    param = model.get_parameter(target_name)
                    param.data.copy_(qkv_weight)
                    qkv_count += 1
                except AttributeError:
                    pass
            else:
                if q_key not in weight_cache:
                    print(f"⚠ Missing {q_key}")
        
        # Fuse gate_up weights
        gate_up_count = 0
        for i in range(num_layers):
            gate_key = f'model.transformer.blocks.{i}.ff_proj.weight'
            up_key = f'model.transformer.blocks.{i}.up_proj.weight'
            
            if gate_key in weight_cache and up_key in weight_cache:
                gate_up_weight = torch.cat([
                    weight_cache[gate_key],
                    weight_cache[up_key]
                ], dim=0)
                
                target_name = f'model.layers.{i}.mlp.gate_up_proj.weight'
                try:
                    param = model.get_parameter(target_name)
                    param.data.copy_(gate_up_weight)
                    gate_up_count += 1
                except AttributeError:
                    pass
            else:
                if gate_key not in weight_cache:
                    print(f"⚠ Missing {gate_key}")
                if up_key not in weight_cache:
                    print(f"⚠ Missing {up_key}")
        
        total_loaded = loaded_count + qkv_count + gate_up_count
        print(f"✓ Weight loading complete:")
        print(f"  - Direct mappings: {loaded_count}")
        print(f"  - QKV fusions: {qkv_count}")
        print(f"  - Gate-Up fusions: {gate_up_count}")
        print(f"  - Total: {total_loaded}\n")
        return
    
    # Original load logic for SDAR
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        if _is_moe_expert_weight(param_name):
                            _load_expert_weight_to_fused(model, param_name, f.get_tensor(weight_name), shard_id)
                        else:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    if _is_moe_expert_weight(weight_name):
                        _load_expert_weight_to_fused(model, weight_name, f.get_tensor(weight_name))
                    else:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))


def load_from_hf_model(target_model: nn.Module, hf_model: nn.Module):
    """
    Load weights from a HuggingFace transformer model to the custom model.
    
    Args:
        target_model: model for inference engine
        hf_model: model from HuggingFace modeling
    """
    import torch
    import torch.nn as nn

    # Get device from HF model
    device = next(hf_model.parameters()).device

    # Track which parameters have been loaded
    loaded_params = set()
    missing_params = []

    # Build a new state dict with materialized tensors
    new_state_dict = {}

    # Process each parameter in the target model
    for name, param in target_model.named_parameters():
        with torch.no_grad():
            # Handle fused qkv_proj
            if 'qkv_proj.weight' in name:
                base_name = name.replace('qkv_proj.weight', '')
                q_name = base_name + 'q_proj.weight'
                k_name = base_name + 'k_proj.weight'
                v_name = base_name + 'v_proj.weight'

                try:
                    q_weight = hf_model.get_parameter(q_name)
                    k_weight = hf_model.get_parameter(k_name)
                    v_weight = hf_model.get_parameter(v_name)

                    # Concatenate Q, K, V weights along dim 0 (output dimension)
                    qkv_weight = torch.cat(
                        [q_weight, k_weight, v_weight], dim=0)

                    # Convert to target dtype and appropriate device
                    if param.is_meta:
                        # Materialize on the HF model's device
                        new_state_dict[name] = qkv_weight.to(
                            device=device, dtype=param.dtype)
                    else:
                        # Keep on parameter's current device
                        new_state_dict[name] = qkv_weight.to(
                            device=param.device, dtype=param.dtype)

                    loaded_params.add(name)
                except AttributeError:
                    missing_params.append(name)
                    # Only keep existing non-meta parameters
                    if not param.is_meta:
                        new_state_dict[name] = param.data

            # Handle fused gate_up_proj (non-MoE)
            elif 'gate_up_proj.weight' in name and 'experts.' not in name:
                base_name = name.replace('gate_up_proj.weight', '')
                gate_name = base_name + 'gate_proj.weight'
                up_name = base_name + 'up_proj.weight'

                try:
                    gate_weight = hf_model.get_parameter(gate_name)
                    up_weight = hf_model.get_parameter(up_name)

                    # Concatenate gate and up weights along dim 0 (output dimension)
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

                    # Convert to target dtype and appropriate device
                    if param.is_meta:
                        # Materialize on the HF model's device
                        new_state_dict[name] = gate_up_weight.to(
                            device=device, dtype=param.dtype)
                    else:
                        # Keep on parameter's current device
                        new_state_dict[name] = gate_up_weight.to(
                            device=param.device, dtype=param.dtype)

                    loaded_params.add(name)
                except AttributeError:
                    missing_params.append(name)
                    # Only keep existing non-meta parameters
                    if not param.is_meta:
                        new_state_dict[name] = param.data

            # Handle regular parameters (direct mapping)
            else:
                try:
                    hf_param = hf_model.get_parameter(name)

                    # Convert to target dtype and appropriate device
                    if param.is_meta:
                        # Materialize on the HF model's device
                        new_state_dict[name] = hf_param.to(
                            device=device, dtype=param.dtype)
                    else:
                        # Keep on parameter's current device
                        new_state_dict[name] = hf_param.to(
                            device=param.device, dtype=param.dtype)

                    loaded_params.add(name)
                except AttributeError:
                    # Try without model prefix if not found
                    if name.startswith('model.'):
                        try:
                            hf_param = hf_model.get_parameter(name[6:])

                            # Convert to target dtype and appropriate device
                            if param.is_meta:
                                # Materialize on the HF model's device
                                new_state_dict[name] = hf_param.to(
                                    device=device, dtype=param.dtype)
                            else:
                                # Keep on parameter's current device
                                new_state_dict[name] = hf_param.to(
                                    device=param.device, dtype=param.dtype)

                            loaded_params.add(name)
                        except AttributeError:
                            missing_params.append(name)
                            # Only keep existing non-meta parameters
                            if not param.is_meta:
                                new_state_dict[name] = param.data
                    else:
                        missing_params.append(name)
                        # Only keep existing non-meta parameters
                        if not param.is_meta:
                            new_state_dict[name] = param.data

    # Load the new state dict into the target model
    # Use strict=False to handle any mismatches gracefully
    target_model.load_state_dict(new_state_dict, assign=True)

    # Disable gradients for all parameters (inference mode)
    for param in target_model.parameters():
        param.requires_grad_(False)

    # Report loading status
    print(
        f"Successfully loaded {len(loaded_params)}/{len(list(target_model.named_parameters()))} parameters")
    if missing_params:
        print(
            f"Warning: Could not find {len(missing_params)} parameters in HF model:")
        for param in missing_params[:10]:  # Show first 10
            print(f"  - {param}")
        if len(missing_params) > 10:
            print(f"  ... and {len(missing_params) - 10} more")

    # Check if there are still meta parameters after loading
    meta_params = [name for name,
                   param in target_model.named_parameters() if param.is_meta]
    if meta_params:
        print(
            f"ERROR: {len(meta_params)} parameters are still on meta device after loading!")
        for param in meta_params[:5]:
            print(f"  - {param}")
        if len(meta_params) > 5:
            print(f"  ... and {len(meta_params) - 5} more")
        raise RuntimeError(
            f"Failed to materialize {len(meta_params)} meta parameters")

