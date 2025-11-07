# 代码修改审查

## Git 状态说明

Git 显示大量 "deleted: BlockInfer/" 是因为**目录重命名**:
- 旧: `BlockInfer/` (大写)
- 新: `blockinfer/` (小写)

这是**正确的改动**，符合 Python 包命名规范。

## 实际修改检查

### 1. config.py ✅ 安全

**原有逻辑**:
```python
assert os.path.isdir(self.model)  # 严格检查必须是目录
```

**新逻辑**:
```python
# 移除严格检查，支持 HuggingFace 模型 ID
# 添加灵活的 max_position 处理
```

**影响**: 
- ✅ SDAR: 本地路径仍然工作
- ✅ LLaDA: 现在可以使用 HF 模型 ID
- ✅ 向后兼容

### 2. model_runner.py ✅ 安全

**添加的代码**:
```python
# Line 12: Import LLaDA
from blockinfer.models.llada import LLaDAForCausalLM

# Line 34-37: 安全的 dtype 处理
model_dtype = getattr(hf_config, 'torch_dtype', torch.bfloat16)
if model_dtype is None or not model_dtype.is_floating_point:
    model_dtype = torch.bfloat16

# Line 40-45: 添加 LLaDA 分支（SDAR 逻辑完全保留）
if "sdar" in hf_config.model_type and "moe" in hf_config.model_type:
    self.model = SDARMoeForCausalLM(hf_config)
elif "sdar" in hf_config.model_type:
    self.model = SDARForCausalLM(hf_config)
elif "llada" in hf_config.model_type.lower():  # ← 新增
    self.model = LLaDAForCausalLM(hf_config)

# Line 141-144: 灵活的配置属性获取
num_kv_heads = getattr(hf_config, 'num_key_value_heads',
                      getattr(hf_config, 'num_kv_heads', 
                             hf_config.num_attention_heads))
```

**影响**:
- ✅ SDAR: 完全不受影响（elif 分支保留）
- ✅ 新增: LLaDA 支持
- ✅ 更健壮: 属性获取不会崩溃

### 3. scheduler.py ✅ 安全

**原有策略（完全保留）**:
```python
Line 130: 'sequential'
Line 136: 'low_confidence_static'
Line 143: 'low_confidence_dynamic'  
Line 152: 'entropy_bounded'
```

**新增策略（LLaDA）**:
```python
Line 168: 'low_confidence'
Line 176: 'random'
```

**影响**:
- ✅ SDAR: 所有原有策略完整保留
- ✅ 新增: 2 个 LLaDA 策略
- ✅ 兼容: elif 结构不冲突

### 4. loader.py ✅ 安全

**添加的代码**:
```python
# Line 123-150: LLaDA 权重映射函数（新函数）
def _create_llada_weight_mapping(num_layers: int):
    ...

# Line 152-178: LLaDA 模型检测（新函数）
def _is_llada_model(model_path: str) -> bool:
    ...

# Line 181-278: LLaDA 权重加载分支（新分支）
def load_model(model, path):
    if is_llada:
        # LLaDA 加载逻辑
        ...
        return
    
    # 原有 SDAR 逻辑（完全保留）
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        ...
```

**影响**:
- ✅ SDAR: 当检测到非 LLaDA 模型时，使用原有逻辑
- ✅ 新增: LLaDA 特殊处理
- ✅ 安全: 提前返回，不影响 SDAR 路径

### 5. sampling_params.py ✅ 安全

**修改**:
```python
# 扩展 remasking_strategy 类型
Literal['sequential', 'low_confidence_static', ..., 'random', 'low_confidence']

# 添加 LLaDA 参数（有默认值）
cfg_scale: float = 0.0
logits_eos_inf: bool = False
confidence_eos_eot_inf: bool = False
```

**影响**:
- ✅ SDAR: 默认值不影响现有用法
- ✅ 新增: LLaDA 特有参数
- ✅ 兼容: 所有字段有默认值

## 向后兼容性验证

### SDAR 代码路径检查

**Model Loading**:
```python
if "sdar" in model_type and "moe": SDARMoeForCausalLM  ✓
elif "sdar" in model_type: SDARForCausalLM             ✓
elif "llada": LLaDAForCausalLM                         ← 新增
```
**结论**: ✅ SDAR 分支完全保留

**Remasking Strategies**:
```python
if 'sequential': ...           ✓ 保留
elif 'low_confidence_static':  ✓ 保留
elif 'low_confidence_dynamic': ✓ 保留
elif 'entropy_bounded':        ✓ 保留
elif 'low_confidence':         ← 新增
elif 'random':                 ← 新增
```
**结论**: ✅ SDAR 策略完全保留

**Weight Loading**:
```python
if is_llada_model:
    # LLaDA loading
    return  ← 提前返回
    
# SDAR loading (原有逻辑)
packed_modules_mapping = ...  ✓ 保留
```
**结论**: ✅ SDAR 加载逻辑完全保留

## 最终结论

### ✅ 所有修改都是**安全的添加**:

1. **config.py**: 移除过严检查 → 更灵活，SDAR 不受影响
2. **model_runner.py**: 添加 LLaDA 分支 → SDAR elif 保留
3. **scheduler.py**: 添加策略 → SDAR 策略完全保留
4. **loader.py**: 添加 LLaDA 路径 → SDAR 路径保留
5. **sampling_params.py**: 添加参数 → 有默认值，不影响 SDAR

### ✅ 目录重命名:

`BlockInfer/` → `blockinfer/` 
- 这是**正确的**，符合 Python 包规范
- 功能完全不受影响
- Git 会正确处理重命名

### 🎯 结论

**所有修改都不会影响原有 SDAR 功能！**

可以安全 push。
