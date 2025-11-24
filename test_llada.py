"""
Test script for LLaDA using the unified BlockInfer API.
LLM class automatically detects and configures for LLaDA models.
"""

from blockinfer import LLM
from blockinfer.sampling_params import SamplingParams
from transformers import AutoTokenizer

from utils import find_cached_model


def main():
    print("="*80)
    print("LLaDA Implementation Test")
    print("Using corrected logic from Fast-dLLM/llada/generate.py")
    print("="*80)
    
    # Point to local cache if present
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    cached_path = find_cached_model(model_id)
    if cached_path:
        model_path = cached_path
        print(f"Found cached model at {model_path}")
    else:
        model_path = model_id
        print(f"Using HF repo id '{model_id}'. Will download if not cached locally.")
    
    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # LLaDA uses mask_token_id=126336
    # The unified LLM class automatically detects LLaDA models and configures accordingly
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        mask_token_id=126336,
        block_length=32
    )
    
    # Sampling parameters matching LLaDA's typical usage
    # Using temperature=0.0 as in Fast-dLLM examples
    sampling_params = SamplingParams(
        temperature=0.0,  # Use greedy decoding like Fast-dLLM
        topk=0,  # Disable top-k when temperature is 0
        topp=1.0,
        max_tokens=128,
        remasking_strategy="low_confidence",  # Match Fast-dLLM's strategy
        block_length=32,
        denoising_steps=32,  # 32 steps per 32-token block
    )
    
    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"
    ]
    
    # Apply chat template if available
    if tokenizer.chat_template:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in prompts
        ]
    else:
        formatted_prompts = prompts
    
    print("\n" + "="*80)
    print("Generating...")
    print("="*80)
    
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for prompt, output in zip(prompts, outputs):
        print("\n" + "="*50)
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")
        print(f"Total tokens: {len(output['token_ids'])}")
        print(f"Token IDs head: {output['token_ids'][:32]}")
        print(f"Token IDs tail: {output['token_ids'][-10:]}")
        print("="*50)
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

