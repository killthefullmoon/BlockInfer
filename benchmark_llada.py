"""
Performance comparison between BlockInfer LLaDA (block diffusion) and Vanilla LLaDA (non-block diffusion).
Uses sample_200.jsonl dataset for testing.
"""

import os
import sys
import json
import time
import re
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from blockinfer import LLM, SamplingParams
from utils import find_cached_model


def format_prompt_for_dataset(question_data):
    """
    Format prompt based on dataset type for better answer extraction.
    
    Args:
        question_data: Dict with 'question', 'source', 'choices' fields
        
    Returns:
        Formatted prompt string
    """
    question = question_data['question']
    source = question_data['source']
    choices = question_data.get('choices', '')
    
    if source == 'gsm8k':
        # Math word problems - request answer in braces
        return f"{question}\n\nPlease solve this step by step and put your final numerical answer in curly braces like {{answer}}."
    
    elif source == 'math500':
        # Advanced math - request answer in braces
        return f"{question}\n\nSolve this problem and put your final answer in curly braces like {{answer}}."
    
    elif source == 'mmlu':
        # Multiple choice - format with choices and request answer in braces
        if choices:
            try:
                choices_list = eval(choices) if isinstance(choices, str) else choices
                formatted_choices = '\n'.join([f"{i}. {choice}" for i, choice in enumerate(choices_list)])
                return f"{question}\n\n{formatted_choices}\n\nPlease select the correct option and put ONLY the number (0, 1, 2, or 3) in curly braces like {{0}}."
            except:
                return f"{question}\n\nPlease put your answer (0-3) in curly braces like {{0}}."
        else:
            return f"{question}\n\nPlease put your answer in curly braces like {{answer}}."
    
    elif source == 'longbench_hotpotqa':
        # QA - request concise answer in braces
        return f"{question}\n\nProvide a concise answer and put it in curly braces like {{answer}}."
    
    else:
        # Default format
        return f"{question}\n\nPut your answer in curly braces like {{answer}}."


def extract_answer(text, source):
    """
    Extract answer from generated text based on source type.
    Priority: {answer} format > fallback patterns
    """
    # First, try to extract from curly braces (our requested format)
    brace_match = re.search(r'\{([^}]+)\}', text)
    if brace_match:
        answer = brace_match.group(1).strip()
        # Clean up the answer
        if source in ['gsm8k', 'math500']:
            # For math, extract just the number/expression
            answer = answer.replace(',', '').strip()
        return answer
    
    # Fallback patterns if model didn't follow instructions
    if source == 'gsm8k':
        # GSM8K: Look for numerical answers
        patterns = [
            r'####\s*([0-9,\.]+)',
            r'[Aa]nswer:\s*([0-9,\.]+)',
            r'[Rr]esult:\s*([0-9,\.]+)',
            r'=\s*([0-9,\.]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(',', '')
        
        # Last resort: extract last number
        numbers = re.findall(r'[0-9,\.]+', text)
        if numbers:
            return numbers[-1].replace(',', '')
    
    elif source == 'math500':
        # Math500: Look for LaTeX boxed or final answer
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'[Aa]nswer:\s*(.+?)(?:\.|<|$)',
            r'[Ff]inal answer:\s*(.+?)(?:\.|<|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    
    elif source == 'mmlu':
        # MMLU: Extract option number (0-3)
        # Look for standalone digits first
        match = re.search(r'(?:^|[^\d])([0-3])(?:[^\d]|$)', text)
        if match:
            return match.group(1)
        # Fallback to any occurrence of 0-3
        match = re.search(r'[0-3]', text)
        if match:
            return match.group(0)
    
    elif source == 'longbench_hotpotqa':
        # For QA, try to extract a concise answer
        patterns = [
            r'[Aa]nswer:\s*(.+?)(?:\.|<|$)',
            r'[Tt]he answer is\s*(.+?)(?:\.|<|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    
    # Ultimate fallback: return first few words
    words = text.split()[:10]
    return ' '.join(words) if words else text[:100]


def calculate_accuracy(outputs, questions):
    """Calculate accuracy by comparing extracted answers with ground truth."""
    correct = 0
    total = 0
    results_detail = []
    
    for output, question in zip(outputs, questions):
        if not question['answer']:
            continue
        
        total += 1
        predicted = extract_answer(output['text'], question['source'])
        ground_truth = str(question['answer']).strip()
        
        # Normalize for comparison
        predicted_norm = predicted.lower().replace(' ', '').replace(',', '')
        ground_truth_norm = ground_truth.lower().replace(' ', '').replace(',', '')
        
        is_correct = predicted_norm == ground_truth_norm
        if is_correct:
            correct += 1
        
        results_detail.append({
            'question': question['question'][:100],
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': is_correct,
            'source': question['source']
        })
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total, results_detail


def load_dataset(dataset_path, num_samples=None, filter_source=None, skip_samples=0):
    """
    Load questions from sample_200.jsonl.
    
    Args:
        dataset_path: Path to the dataset file
        num_samples: Maximum number of samples to load (per source if filtering)
        filter_source: Filter by source type (e.g., 'gsm8k', 'mmlu', 'math500', 'longbench_hotpotqa')
        skip_samples: Number of samples to skip at the beginning
    """
    questions = []
    skipped = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Apply source filter if specified
            if filter_source and data.get('source', '') != filter_source:
                continue
            
            # Skip first N samples
            if skipped < skip_samples:
                skipped += 1
                continue
            
            questions.append({
                'question': data['question'],
                'answer': data.get('answer', ''),
                'source': data.get('source', ''),
                'choices': data.get('choices', '')
            })
            if num_samples and len(questions) >= num_samples:
                break
    return questions


def benchmark_blockinfer(model_path, questions, sampling_params, tokenizer):
    """Benchmark BlockInfer block-wise diffusion implementation."""
    print("\n" + "="*80)
    print("BENCHMARKING: BlockInfer (Block-wise Diffusion)")
    print("="*80)
    
    # Initialize BlockInfer LLM
    print("Loading BlockInfer LLM...")
    tp_size_env = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    if tp_size_env > 1:
        print(f"TENSOR_PARALLEL_SIZE={tp_size_env} requested, but LLaDA backend only supports single-process; falling back to 1.")
    tp_size = 1
    print(f"Using tensor parallel size: {tp_size}")
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        mask_token_id=126336,
        block_length=sampling_params.block_length
    )
    
    # Prepare prompts with dataset-specific formatting
    formatted_questions = [format_prompt_for_dataset(q) for q in questions]
    
    # Apply chat template
    if tokenizer.chat_template:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in formatted_questions
        ]
    else:
        formatted_prompts = formatted_questions
    
    # Warm up
    print("Warming up...")
    _ = llm.generate([formatted_prompts[0]], sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark on {len(questions)} questions...")
    start_time = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    total_tokens = sum(len(out['token_ids']) for out in outputs)
    throughput = total_tokens / total_time
    avg_latency = total_time / len(questions)
    
    results = {
        'implementation': 'BlockInfer (Block Diffusion)',
        'total_time': total_time,
        'total_tokens': total_tokens,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'num_prompts': len(questions),
        'outputs': outputs,
        'questions': questions
    }
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Avg latency: {avg_latency:.3f}s per question")
    print(f"  Avg tokens per question: {total_tokens/len(questions):.1f}")
    
    return results


def benchmark_vanilla_llada(model_path, questions, max_tokens, tokenizer):
    """Benchmark vanilla LLaDA (non-block diffusion, all tokens at once)."""
    print("\n" + "="*80)
    print("BENCHMARKING: Vanilla LLaDA (Non-Block Diffusion)")
    print("="*80)
    vanilla_dp_size = max(1, int(os.environ.get("VANILLA_DATA_PARALLEL_SIZE", "1")))
    available_gpus = torch.cuda.device_count()
    device_ids = list(range(min(vanilla_dp_size, available_gpus))) if available_gpus > 0 else []
    primary_device = torch.device(f"cuda:{device_ids[0]}") if device_ids else torch.device("cpu")
    
    # Import the original LLaDA generate function
    llada_path = Path(__file__).parent.parent / "LLaDA"
    if llada_path.exists():
        sys.path.insert(0, str(llada_path))
        from generate import generate
    else:
        print(f"Error: Cannot find LLaDA directory at {llada_path}")
        print("Please ensure LLaDA is in the parent directory.")
        return None
    
    print("Loading vanilla LLaDA model...")
    # Use AutoModel as in original LLaDA example
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(primary_device).eval()
    
    if len(device_ids) > 1:
        print(f"Enabling DataParallel for vanilla LLaDA on devices: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        print("Running vanilla LLaDA on a single GPU")
    
    # Prepare prompts with dataset-specific formatting
    formatted_questions = [format_prompt_for_dataset(q) for q in questions]
    
    # Warm up
    print("Warming up...")
    test_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted_questions[0]}],
        add_generation_prompt=True,
        tokenize=False
    )
    test_input_ids = torch.tensor([tokenizer.encode(test_prompt)]).to(primary_device)
    
    # Vanilla LLaDA: block_length = gen_length (all tokens at once, no block division)
    # This is the KEY difference from BlockInfer
    _ = generate(
        model,
        test_input_ids,
        steps=max_tokens,  # steps = gen_length for non-block diffusion
        gen_length=max_tokens,
        block_length=max_tokens,  # KEY: block_length = gen_length (no blocking)
        temperature=0.,
        remasking='low_confidence',
        mask_id=126336
    )
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark on {len(questions)} questions...")
    start_time = time.time()
    outputs = []
    
    for i, formatted_question in enumerate(formatted_questions):
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_question}],
            add_generation_prompt=True,
            tokenize=False
        )
        input_ids = torch.tensor([tokenizer.encode(formatted_prompt)]).to(primary_device)
        
        # Vanilla LLaDA: Non-block diffusion (all tokens at once)
        # block_length = gen_length means no block division
        output_ids = generate(
            model,
            input_ids,
            steps=max_tokens,  # steps = gen_length
            gen_length=max_tokens,
            block_length=max_tokens,  # No block division
            temperature=0.,
            remasking='low_confidence',
            mask_id=126336
        )
        
        # Extract completion
        completion_ids = output_ids[0, input_ids.shape[1]:].cpu().tolist()
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        outputs.append({
            'text': completion_text,
            'token_ids': completion_ids
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{len(questions)} questions")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    total_tokens = sum(len(out['token_ids']) for out in outputs)
    throughput = total_tokens / total_time
    avg_latency = total_time / len(questions)
    
    results = {
        'implementation': 'Vanilla LLaDA (Non-Block Diffusion)',
        'total_time': total_time,
        'total_tokens': total_tokens,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'num_prompts': len(questions),
        'outputs': outputs,
        'questions': questions
    }
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Avg latency: {avg_latency:.3f}s per question")
    print(f"  Avg tokens per question: {total_tokens/len(questions):.1f}")
    
    return results


def compare_results(blockinfer_results, vanilla_results):
    """Compare and display results from both implementations."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if vanilla_results is None:
        print("Vanilla LLaDA results not available for comparison.")
        print("\nBlockInfer standalone results:")
        print(f"  Throughput: {blockinfer_results['throughput']:.2f} tokens/s")
        print(f"  Total time: {blockinfer_results['total_time']:.2f}s")
        print(f"  Avg latency: {blockinfer_results['avg_latency']:.3f}s per question")
        return
    
    print(f"\n{'Metric':<30} {'BlockInfer':<20} {'Vanilla LLaDA':<20} {'Speedup':<15}")
    print("-" * 85)
    
    # Throughput
    bi_throughput = blockinfer_results['throughput']
    vl_throughput = vanilla_results['throughput']
    speedup = bi_throughput / vl_throughput
    print(f"{'Throughput (tokens/s)':<30} {bi_throughput:>18.2f} {vl_throughput:>18.2f} {speedup:>13.2f}x")
    
    # Total time
    bi_time = blockinfer_results['total_time']
    vl_time = vanilla_results['total_time']
    speedup = vl_time / bi_time
    print(f"{'Total time (s)':<30} {bi_time:>18.2f} {vl_time:>18.2f} {speedup:>13.2f}x")
    
    # Average latency
    bi_latency = blockinfer_results['avg_latency']
    vl_latency = vanilla_results['avg_latency']
    speedup = vl_latency / bi_latency
    print(f"{'Avg latency (s/question)':<30} {bi_latency:>18.3f} {vl_latency:>18.3f} {speedup:>13.2f}x")
    
    # Total tokens
    bi_tokens = blockinfer_results['total_tokens']
    vl_tokens = vanilla_results['total_tokens']
    print(f"{'Total tokens generated':<30} {bi_tokens:>18} {vl_tokens:>18} {'':<15}")
    
    # Average tokens per question
    bi_avg_tokens = bi_tokens / blockinfer_results['num_prompts']
    vl_avg_tokens = vl_tokens / vanilla_results['num_prompts']
    print(f"{'Avg tokens per question':<30} {bi_avg_tokens:>18.1f} {vl_avg_tokens:>18.1f} {'':<15}")
    
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS (First 3 Questions)")
    print("="*80)
    
    for i in range(min(3, len(blockinfer_results['questions']))):
        print(f"\n{'='*60}")
        print(f"Question {i+1} ({blockinfer_results['questions'][i]['source']}):")
        print(f"{'='*60}")
        print(f"Q: {blockinfer_results['questions'][i]['question'][:150]}...")
        if blockinfer_results['questions'][i]['answer']:
            print(f"Expected: {blockinfer_results['questions'][i]['answer']}")
        
        print(f"\nBlockInfer Output:")
        print(blockinfer_results['outputs'][i]['text'][:200])
        
        if vanilla_results:
            print(f"\nVanilla LLaDA Output:")
            print(vanilla_results['outputs'][i]['text'])


def save_results(blockinfer_results, vanilla_results, output_file="benchmark_results.json"):
    """Save benchmark results to file."""
    results = {
        'blockinfer': {
            'total_time': blockinfer_results['total_time'],
            'total_tokens': blockinfer_results['total_tokens'],
            'throughput': blockinfer_results['throughput'],
            'avg_latency': blockinfer_results['avg_latency'],
            'num_prompts': blockinfer_results['num_prompts'],
            'accuracy': blockinfer_results.get('accuracy', 0),
            'correct': blockinfer_results.get('correct', 0),
            'total': blockinfer_results.get('total', 0)
        }
    }
    
    if vanilla_results:
        results['vanilla_llada'] = {
            'total_time': vanilla_results['total_time'],
            'total_tokens': vanilla_results['total_tokens'],
            'throughput': vanilla_results['throughput'],
            'avg_latency': vanilla_results['avg_latency'],
            'num_prompts': vanilla_results['num_prompts'],
            'accuracy': vanilla_results.get('accuracy', 0),
            'correct': vanilla_results.get('correct', 0),
            'total': vanilla_results.get('total', 0)
        }
        results['comparison'] = {
            'throughput_speedup': blockinfer_results['throughput'] / vanilla_results['throughput'],
            'time_speedup': vanilla_results['total_time'] / blockinfer_results['total_time'],
            'accuracy_diff': blockinfer_results.get('accuracy', 0) - vanilla_results.get('accuracy', 0)
        }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def main():
    """Run performance comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark LLaDA: Block vs Non-Block Diffusion')
    parser.add_argument('--dataset', type=str, default='sample_200.jsonl',
                        help='Path to dataset file (default: sample_200.jsonl)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to use (default: all)')
    parser.add_argument('--source', type=str, default=None,
                        choices=['gsm8k', 'math500', 'mmlu', 'longbench_hotpotqa'],
                        help='Filter by data source (default: None, use all sources)')
    parser.add_argument('--max-tokens', type=int, default=128,
                        help='Maximum tokens to generate (default: 128)')
    parser.add_argument('--block-length', type=int, default=32,
                        help='Block length for BlockInfer (default: 32)')
    parser.add_argument('--denoising-steps', type=int, default=32,
                        help='Denoising steps per block for BlockInfer (default: 32)')
    parser.add_argument('--skip-vanilla', action='store_true',
                        help='Skip Vanilla LLaDA benchmark')
    parser.add_argument('--skip-sample', type=int, default=0,
                        help='Skip first N samples (for sequential processing)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("LLaDA Performance Comparison")
    print("Block Diffusion (BlockInfer) vs Autoregressive Decoding (Vanilla LLaDA)")
    print("="*80)
    
    # Find cached model
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    cached_path = find_cached_model(model_id)
    if cached_path:
        model_path = cached_path
        print(f"\nFound cached model at {model_path}")
    else:
        model_path = model_id
        print(f"\nUsing HF repo id '{model_id}'. Will download if not cached locally.")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    if args.source:
        print(f"Filtering by source: {args.source}")
    if args.skip_sample > 0:
        print(f"Skipping first {args.skip_sample} samples")
    questions = load_dataset(args.dataset, args.num_samples, args.source, args.skip_sample)
    print(f"Loaded {len(questions)} questions")
    
    # Show dataset statistics
    source_counts = {}
    for q in questions:
        source = q['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nDataset composition:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} questions")
    
    # Sampling parameters for BlockInfer
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        topk=0,           # Disable top-k for greedy
        topp=1.0,         # Disable top-p for greedy
        max_tokens=args.max_tokens,
        remasking_strategy="low_confidence",
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        ignore_eos=True,  # ⚠️ Force generation of max_tokens for fair comparison
    )
    
    print(f"\nBenchmark configuration:")
    print(f"  Number of questions: {len(questions)}")
    if args.source:
        print(f"  Data source: {args.source} only")
    else:
        print(f"  Data source: all sources")
    print(f"  Max tokens per question: {args.max_tokens}")
    print(f"  Ignore EOS: True (force both implementations to generate exactly {args.max_tokens} tokens)")
    print(f"  BlockInfer: block_length={args.block_length}, steps={args.denoising_steps} (Block Diffusion)")
    print(f"  Vanilla LLaDA: block_length={args.max_tokens}, steps={args.max_tokens} (Non-Block Diffusion)")
    
    # Benchmark BlockInfer
    blockinfer_results = benchmark_blockinfer(
        model_path,
        questions,
        sampling_params,
        tokenizer
    )
    
    # Benchmark Vanilla LLaDA (optional)
    vanilla_results = None
    if not args.skip_vanilla:
        vanilla_results = benchmark_vanilla_llada(
            model_path,
            questions,
            args.max_tokens,
            tokenizer
        )
    
    # Calculate accuracy
    print("\n" + "="*80)
    print("ACCURACY EVALUATION")
    print("="*80)
    
    bi_accuracy, bi_correct, bi_total, bi_details = calculate_accuracy(
        blockinfer_results['outputs'],
        blockinfer_results['questions']
    )
    print(f"\nBlockInfer Accuracy: {bi_correct}/{bi_total} = {bi_accuracy:.1f}%")
    blockinfer_results['accuracy'] = bi_accuracy
    blockinfer_results['correct'] = bi_correct
    blockinfer_results['total'] = bi_total
    
    if vanilla_results:
        vl_accuracy, vl_correct, vl_total, vl_details = calculate_accuracy(
            vanilla_results['outputs'],
            vanilla_results['questions']
        )
        print(f"Vanilla LLaDA Accuracy: {vl_correct}/{vl_total} = {vl_accuracy:.1f}%")
        print(f"Accuracy Difference: {bi_accuracy - vl_accuracy:+.1f}%")
        vanilla_results['accuracy'] = vl_accuracy
        vanilla_results['correct'] = vl_correct
        vanilla_results['total'] = vl_total
        
        # Show some examples
        print(f"\nSample Predictions (First 3):")
        for i in range(min(3, len(bi_details))):
            print(f"\n{'='*60}")
            print(f"Q: {bi_details[i]['question']}...")
            print(f"Ground Truth: {bi_details[i]['ground_truth']}")
            print(f"BlockInfer: {bi_details[i]['predicted']} {'✓' if bi_details[i]['correct'] else '✗'}")
            print(f"Vanilla: {vl_details[i]['predicted']} {'✓' if vl_details[i]['correct'] else '✗'}")
    
    # Compare results
    compare_results(blockinfer_results, vanilla_results)
    
    # Save results
    save_results(blockinfer_results, vanilla_results, args.output)
    
    print("\n" + "="*80)
    print("Benchmark completed!")
    print("="*80)


if __name__ == "__main__":
    main()
