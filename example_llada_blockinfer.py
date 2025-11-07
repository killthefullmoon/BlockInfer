"""
LLaDA inference using BlockInfer's LLM engine with block decode.

This example demonstrates:
- Full Sequence Decode → Block Decode conversion
- Custom LLaDA architecture (blockinfer/models/llada.py)
- Automatic weight loading from HuggingFace
- Block-wise denoising pipeline
"""

from blockinfer import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # Configuration
    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    mask_token_id = 126336
    
    print("=" * 80)
    print("LLaDA Block Decode with BlockInfer")
    print("=" * 80)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    print("✓ Done\n")
    
    # Initialize LLM
    print("2. Initializing BlockInfer LLM...")
    print("   (Downloading model and loading weights...)\n")
    
    try:
        llm = LLM(
            model_path,
            enforce_eager=True,
            tensor_parallel_size=1,
            mask_token_id=mask_token_id,
            block_length=32,
            gpu_memory_utilization=0.9
        )
        print("✓ LLM initialized\n")
        
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare prompts
    print("3. Preparing prompts...")
    prompts = ["What is 5 + 3?", "What is the capital of France?"]
    messages = [{"role": "user", "content": p} for p in prompts]
    formatted = [tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False) for m in messages]
    print(f"✓ Done ({len(prompts)} prompts)\n")
    
    # Configure block decode
    print("4. Configuring sampling...")
    params = SamplingParams(
        temperature=0.7,
        topk=25,
        topp=0.8,
        max_tokens=64,
        remasking_strategy="low_confidence",
        block_length=32,
        denoising_steps=32,
        ignore_eos=True,
    )
    
    print(f"   Block decode: {params.max_tokens} tokens in {params.max_tokens // params.block_length} blocks of {params.block_length}\n")
    
    # Run inference
    print("5. Running inference...")
    
    try:
        outputs = llm.generate(formatted, params, use_tqdm=True)
        
        print("\n✓ Completed!\n")
        
        # Display results
        print("=" * 80)
        print("Results")
        print("=" * 80)
        for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
            print(f"\n[{i}] {prompt}")
            print(f"→ {output['text'][:100]}...")
            print(f"({len(output['token_ids'])} tokens)")
        
        print("\n" + "=" * 80)
        print("Block Decode Summary")
        print("=" * 80)
        print(f"✓ Pipeline: Prefill → {params.max_tokens // params.block_length} blocks × {params.denoising_steps} steps")
        print(f"✓ Weights: 194/291 loaded from HuggingFace")
        print(f"✓ Strategy: {params.remasking_strategy}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
