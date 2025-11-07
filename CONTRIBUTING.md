# Contributing to BlockInfer

## Adding New Model Support

BlockInfer supports block diffusion models. To add a new model:

1. **Create model architecture** in `blockinfer/models/your_model.py`
   - Implement forward pass compatible with BlockInfer interface
   - Use existing layers (attention, linear, etc.)

2. **Add model type detection** in `blockinfer/engine/model_runner.py`
   ```python
   elif "your_model" in hf_config.model_type.lower():
       self.model = YourModelForCausalLM(hf_config)
   ```

3. **Add weight loading** (if needed) in `blockinfer/utils/loader.py`
   - Create weight mapping for parameter name differences
   - Handle weight fusion if applicable

4. **Test** with example script

## LLaDA Implementation Example

See the LLaDA integration as a reference:
- Model: `blockinfer/models/llada.py`
- Attention: `blockinfer/layers/llada_attention.py`
- Weight loading: `blockinfer/utils/loader.py` (_is_llada_model, _create_llada_weight_mapping)
- Example: `example_llada_blockinfer.py`

## Code Style

- Follow existing code style
- Add docstrings for public APIs
- Keep functions focused and small
- Comment complex logic

## Testing

Test your changes don't break existing functionality:
```bash
python example.py  # Test SDAR
python example_llada_blockinfer.py  # Test LLaDA
```
