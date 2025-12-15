"""
Performance comparison between BlockInfer SDAR (block diffusion) and Vanilla SDAR (native diffusion_generate).
Uses sample_200.jsonl dataset for testing.
"""

import os
import json
import time
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from blockinfer import LLM, SamplingParams


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


def load_dataset(dataset_path, num_samples=None, filter_source=None):
    """
    Load questions from sample_200.jsonl.
    
    Args:
        dataset_path: Path to the dataset file
        num_samples: Maximum number of samples to load (per source if filtering)
        filter_source: Filter by source type (e.g., 'gsm8k', 'mmlu', 'math500', 'longbench_hotpotqa')
    """
    questions = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Apply source filter if specified
            if filter_source and data.get('source', '') != filter_source:
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


def benchmark_blockinfer_sdar(model_path, questions, sampling_params, tokenizer, use_streaming=False, max_active=1):
    """Benchmark SDAR model using BlockInfer (block-wise diffusion)."""
    print("\n" + "="*80)
    print("BENCHMARKING: BlockInfer SDAR (Block-wise Diffusion)")
    print("="*80)
    
    # Initialize BlockInfer LLM for SDAR
    print("Loading SDAR model with BlockInfer...")
    llm = LLM(
        model_path,
        enforce_eager=False,
        tensor_parallel_size=1,
        mask_token_id=151669,  # SDAR mask token id
        block_length=sampling_params.block_length
    )
    
    # Prepare prompts with dataset-specific formatting
    formatted_questions = [format_prompt_for_dataset(q) for q in questions]
    
    # Apply chat template with enable_thinking=True for SDAR
    if tokenizer.chat_template:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True  # Enable thinking for SDAR
            )
            for prompt in formatted_questions
        ]
    else:
        formatted_prompts = formatted_questions
    
    # Warm up
    print("Warming up...")
    if use_streaming:
        _ = llm.generate_streaming([formatted_prompts[0]], sampling_params, max_active=max_active)
    else:
        _ = llm.generate([formatted_prompts[0]], sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark on {len(questions)} questions...")
    start_time = time.time()
    if use_streaming:
        outputs = llm.generate_streaming(formatted_prompts, sampling_params, max_active=max_active)
    else:
        outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    total_tokens = sum(len(out['token_ids']) for out in outputs)
    throughput = total_tokens / total_time
    avg_latency = total_time / len(questions)
    
    results = {
        'implementation': 'BlockInfer SDAR (Block Diffusion)',
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
    
    # Clean up to free GPU memory for vanilla benchmark
    del llm
    torch.cuda.empty_cache()
    
    return results


def benchmark_vanilla_sdar(
    model_path,
    questions,
    max_tokens,
    tokenizer,
    steps=32,
    block_length=32,
    temperature=0.2,
    top_k=0,
    top_p=1.0,
    remasking_strategy='entropy_bounded',
    confidence_threshold=0.85,
    eb_threshold=0.6,
):
    """Benchmark vanilla SDAR using block_diffusion_generate (non-BlockInfer implementation)."""
    print("\n" + "="*80)
    print("BENCHMARKING: Vanilla SDAR (block_diffusion_generate)")
    print("="*80)
    
    # Patch transformers.utils.LossKwargs to fix ImportError in remote code
    import transformers.utils
    from typing import TypedDict
    
    if not hasattr(transformers.utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            pass
        transformers.utils.LossKwargs = LossKwargs

    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from torch.nn import functional as F
    
    # Helper functions for vanilla generation
    def top_k_logits(logits, k):
        if k <= 0:
            return logits
        else:
            values, _ = torch.topk(logits, k)
            min_values = values[..., -1, None]
            return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

    def top_p_logits(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                     -1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask_indices, float('-inf'))
        return logits

    def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            logits = top_k_logits(logits, top_k)
        if top_p < 1.0:
            logits = top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        assert probs.dim() == 2
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    def get_num_transfer_tokens(blk_length, num_steps):
        base = blk_length // num_steps
        remainder = blk_length % num_steps
        num_transfer_tokens = torch.zeros(num_steps, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    @torch.no_grad()
    def block_diffusion_generate(
            model,
            prompt,
            mask_id,
            gen_length=128,
            blk_length=32,
            denoising_steps=32,
            temp=1.0,
            topk=0,
            topp=1.0,
            remask_strategy='entropy_bounded',
            conf_threshold=0.85,
            eb_thresh=0.6,
            stopping_criteria_idx=None
        ):
        model.eval()
        input_ids = prompt['input_ids']
        prompt_length = input_ids.shape[1]
        past_key_values = DynamicCache()

        num_blocks = (prompt_length + gen_length + blk_length - 1) // blk_length
        total_length = num_blocks * blk_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
        block_diffusion_attention_mask = block_mask.repeat_interleave(blk_length, dim=0)\
                                                   .repeat_interleave(blk_length, dim=1).unsqueeze(0)
        position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
        x[:, :prompt_length] = input_ids
        prefill_blocks = prompt_length // blk_length
        prefill_length = prefill_blocks * blk_length

        # Prefill stage
        if prefill_length > 0:
            cur_x = x[:, :prefill_length]
            cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
            cur_position_ids = position_ids[:, :prefill_length]
            model(cur_x,
                  attention_mask=cur_attn_mask,
                  position_ids=cur_position_ids,
                  past_key_values=past_key_values,
                  use_cache=True,
                  store_kv=True)

        num_transfer_tokens = get_num_transfer_tokens(blk_length, denoising_steps)

        # Decode stage
        for num_block in range(prefill_blocks, num_blocks):
            cur_x = x[:, num_block*blk_length:(num_block+1)*blk_length].clone()
            cur_attn_mask = block_diffusion_attention_mask[
                :, num_block*blk_length:(num_block+1)*blk_length, :(num_block+1)*blk_length
            ]
            cur_position_ids = position_ids[:, num_block*blk_length:(num_block+1)*blk_length]
            
            for step in range(denoising_steps + 1):
                mask_index = (cur_x == mask_id)
                if mask_index.sum() == 0:
                    model(cur_x,
                          attention_mask=cur_attn_mask,
                          position_ids=cur_position_ids,
                          past_key_values=past_key_values,
                          use_cache=True,
                          store_kv=True)
                    break

                logits = model(cur_x,
                               attention_mask=cur_attn_mask,
                               position_ids=cur_position_ids,
                               past_key_values=past_key_values,
                               use_cache=True,
                               store_kv=False).logits

                x0, x0_p = sample_with_temperature_topk_topp(logits, temperature=temp, top_k=topk, top_p=topp)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                
                if remask_strategy == 'sequential':
                    for j in range(cur_x.shape[0]):
                        if mask_index[j].any():
                            first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                            transfer_index[j, first_mask_index:first_mask_index + num_transfer_tokens[step]] = True

                elif remask_strategy == 'low_confidence_static':
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    for j in range(confidence.shape[0]):
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True

                elif remask_strategy == 'low_confidence_dynamic':
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    for j in range(confidence.shape[0]):
                        high_conf_mask = confidence[j] > conf_threshold
                        num_high_confidence = high_conf_mask.sum()
                        if num_high_confidence >= num_transfer_tokens[step]:
                            transfer_index[j] = high_conf_mask
                        else:
                            _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                            transfer_index[j, idx] = True

                elif remask_strategy == "entropy_bounded":
                    # Use full probability distribution for entropy calculation
                    probs = F.softmax(logits, dim=-1)
                    eps = 1e-12
                    # Compute entropy: H = -sum(p * log(p))
                    entropies = -(probs * (probs + eps).log()).sum(dim=-1)  # Shape: [batch, seq_len]
                    entropies = torch.where(mask_index, entropies, torch.inf)
                    ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                    cumsum = torch.cumsum(ent_sorted, dim=1)
                    for j in range(logits.shape[0]):
                        k = torch.searchsorted(cumsum[j], torch.tensor(eb_thresh, device=logits.device), right=False).item()
                        k = max(1, min(k, int(mask_index[j].sum().item())))
                        selected_token_indices = order[j, :k]
                        transfer_index[j, selected_token_indices] = True
                else:
                    raise ValueError(f"Unknown remasking strategy: {remask_strategy}")

                cur_x[transfer_index] = x0[transfer_index]

            x[:, num_block*blk_length:(num_block+1)*blk_length] = cur_x
            if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
                break

        return x

    print("Loading vanilla SDAR model (AutoModelForCausalLM)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    
    mask_id = 151669  # SDAR mask token id
    
    # Prepare prompts with dataset-specific formatting
    formatted_questions = [format_prompt_for_dataset(q) for q in questions]
    
    # Warm up
    print("Warming up...")
    test_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted_questions[0]}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True
    )
    test_tokens = tokenizer(test_prompt, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
    test_tokens = {k: v.to(model.device) for k, v in test_tokens.items()}
    
    _ = block_diffusion_generate(
        model,
        prompt=test_tokens,
        mask_id=mask_id,
        gen_length=min(max_tokens, 64),
        blk_length=block_length,
        denoising_steps=steps,
        temp=temperature,
        topk=top_k,
        topp=top_p,
        remask_strategy=remasking_strategy,
        conf_threshold=confidence_threshold,
        eb_thresh=eb_threshold,
    )
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark on {len(questions)} questions...")
    start_time = time.time()
    outputs = []
    
    for i, formatted_question in enumerate(formatted_questions):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_question}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True
        )
        tokens = tokenizer(formatted_prompt, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        prompt_length = tokens['input_ids'].shape[1]
        
        output_ids = block_diffusion_generate(
            model,
            prompt=tokens,
            mask_id=mask_id,
            gen_length=max_tokens,
            blk_length=block_length,
            denoising_steps=steps,
            temp=temperature,
            topk=top_k,
            topp=top_p,
            remask_strategy=remasking_strategy,
            conf_threshold=confidence_threshold,
            eb_thresh=eb_threshold,
        )
        
        completion_ids = output_ids[0, prompt_length:].cpu().tolist()
        # Remove mask tokens from output
        completion_ids = [t for t in completion_ids if t != mask_id]
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
        'implementation': 'Vanilla SDAR (block_diffusion_generate)',
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
    
    # Clean up to free GPU memory
    del model
    torch.cuda.empty_cache()
    
    return results


def compare_results(blockinfer_results, vanilla_results):
    """Compare and display results from both implementations."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if vanilla_results is None:
        print("Vanilla SDAR results not available for comparison.")
        print("\nBlockInfer standalone results:")
        print(f"  Throughput: {blockinfer_results['throughput']:.2f} tokens/s")
        print(f"  Total time: {blockinfer_results['total_time']:.2f}s")
        print(f"  Avg latency: {blockinfer_results['avg_latency']:.3f}s per question")
        return
    
    if blockinfer_results is None:
        print("BlockInfer SDAR results not available for comparison.")
        print("\nVanilla standalone results:")
        print(f"  Throughput: {vanilla_results['throughput']:.2f} tokens/s")
        print(f"  Total time: {vanilla_results['total_time']:.2f}s")
        print(f"  Avg latency: {vanilla_results['avg_latency']:.3f}s per question")
        return
    
    print(f"\n{'Metric':<30} {'BlockInfer SDAR':<20} {'Vanilla SDAR':<20} {'Speedup':<15}")
    print("-" * 85)
    
    # Throughput
    bi_throughput = blockinfer_results['throughput']
    vl_throughput = vanilla_results['throughput']
    speedup = bi_throughput / vl_throughput if vl_throughput > 0 else float('inf')
    print(f"{'Throughput (tokens/s)':<30} {bi_throughput:>18.2f} {vl_throughput:>18.2f} {speedup:>13.2f}x")
    
    # Total time
    bi_time = blockinfer_results['total_time']
    vl_time = vanilla_results['total_time']
    speedup = vl_time / bi_time if bi_time > 0 else float('inf')
    print(f"{'Total time (s)':<30} {bi_time:>18.2f} {vl_time:>18.2f} {speedup:>13.2f}x")
    
    # Average latency
    bi_latency = blockinfer_results['avg_latency']
    vl_latency = vanilla_results['avg_latency']
    speedup = vl_latency / bi_latency if bi_latency > 0 else float('inf')
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
        
        print(f"\nBlockInfer SDAR Output:")
        print(blockinfer_results['outputs'][i]['text'][:300])
        
        if vanilla_results:
            print(f"\nVanilla SDAR Output:")
            print(vanilla_results['outputs'][i]['text'][:300])


def save_results(blockinfer_results, vanilla_results, output_file="benchmark_sdar_results.json"):
    """Deprecated single-run saver kept for backward compatibility."""
    results_by_source = {"all": {"blockinfer_sdar": blockinfer_results, "vanilla_sdar": vanilla_results}}
    save_results_multi(results_by_source, output_file)


def _pack_outputs(result):
    """Compact outputs to avoid giant files."""
    if result is None:
        return None
    packed_outputs = []
    for i, (out, q) in enumerate(zip(result.get("outputs", []), result.get("questions", []))):
        packed_outputs.append(
            {
                "index": i,
                "question": q.get("question"),
                "ground_truth": q.get("answer"),
                "source": q.get("source"),
                "generated_text": out.get("text"),
                "num_tokens": len(out.get("token_ids", [])),
            }
        )
    compact = result.copy()
    compact["outputs"] = packed_outputs
    compact.pop("questions", None)
    return compact


def save_results_multi(results_by_source, output_file="benchmark_sdar_results.json"):
    """Save benchmark results for multiple sources."""
    output = {}
    for source, results in results_by_source.items():
        bi = results.get("blockinfer_sdar")
        vl = results.get("vanilla_sdar")
        entry = {}
        if bi:
            entry["blockinfer_sdar"] = _pack_outputs(bi)
        if vl:
            entry["vanilla_sdar"] = _pack_outputs(vl)
        if bi and vl:
            entry["comparison"] = {
                "throughput_speedup": bi["throughput"] / vl["throughput"] if vl["throughput"] > 0 else 0,
                "time_speedup": vl["total_time"] / bi["total_time"] if bi["total_time"] > 0 else 0,
                "accuracy_diff": bi.get("accuracy", 0) - vl.get("accuracy", 0),
            }
        output[source] = entry

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")


def main():
    """Run SDAR performance comparison: BlockInfer vs Vanilla."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark SDAR: BlockInfer vs Vanilla diffusion_generate')
    parser.add_argument('--model-path', type=str, 
                        default='models--JetLM--SDAR-4B-Chat',
                        help='Path to SDAR model')
    parser.add_argument('--dataset', type=str, default='sample_200.jsonl',
                        help='Path to dataset file (default: sample_200.jsonl)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to use (default: all)')
    parser.add_argument('--sources', type=str, default='gsm8k,math500,mmlu,longbench_hotpotqa',
                        help='Comma-separated list of sources to benchmark (default: four datasets)')
    parser.add_argument('--dataset-name', type=str, choices=['gsm8k', 'math500', 'mmlu', 'longbench_hotpotqa'],
                        help='Select a single dataset source to benchmark (overrides --sources)')
    parser.add_argument('--max-tokens', type=int, default=256,
                        help='Maximum tokens to generate (default: 128)')
    parser.add_argument('--block-length', type=int, default=4,
                        help='Block length for BlockInfer (default: 4)')
    parser.add_argument('--denoising-steps', type=int, default=4,
                        help='Denoising steps per block (both BlockInfer and vanilla)')
    parser.add_argument('--eb-threshold', type=float, default=0.6,
                        help='Entropy bounded threshold (default: 0.6)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Sampling temperature (default: 0.2)')
    parser.add_argument('--use-streaming', action='store_true',
                        help='Use streaming generation for BlockInfer')
    parser.add_argument('--max-active', type=int, default=1,
                        help='Max active sequences for streaming (default: 1)')
    parser.add_argument('--skip-vanilla', action='store_true',
                        help='Skip Vanilla SDAR benchmark')
    parser.add_argument('--skip-blockinfer', action='store_true',
                        help='Skip BlockInfer SDAR benchmark')
    parser.add_argument('--output', type=str, default='benchmark_sdar_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SDAR Performance Comparison")
    print("BlockInfer (Block Diffusion) vs Vanilla (Native diffusion_generate)")
    print("="*80)
    
    # Model path
    model_path = os.path.expanduser(args.model_path)
    print(f"\nModel path: {model_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if args.dataset_name:
        sources = [args.dataset_name]
    else:
        sources = [s.strip() for s in args.sources.split(',') if s.strip()]
        if not sources:
            sources = ['gsm8k', 'math500', 'mmlu', 'longbench_hotpotqa']

    # Sampling parameters for BlockInfer SDAR
    sampling_params = SamplingParams(
        temperature=args.temperature,
        topk=0,
        topp=1.0,
        max_tokens=args.max_tokens,
        remasking_strategy="entropy_bounded",
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        eb_threshold=args.eb_threshold,
    )
    
    results_by_source = {}
    for src in sources:
        print(f"\n=== Source: {src} ===")
        questions = load_dataset(args.dataset, args.num_samples, src)
        if not questions:
            print(f"  No questions for source '{src}', skipping.")
            continue

        print(f"  Loaded {len(questions)} questions")
        print(f"  block_length={args.block_length}, steps={args.denoising_steps}, max_tokens={args.max_tokens}")

        bi_results = None
        vl_results = None
        bi_details = None
        vl_details = None

        if not args.skip_blockinfer:
            bi_results = benchmark_blockinfer_sdar(
                model_path,
                questions,
                sampling_params,
                tokenizer,
                use_streaming=args.use_streaming,
                max_active=args.max_active,
            )
            bi_accuracy, bi_correct, bi_total, bi_details = calculate_accuracy(
                bi_results["outputs"],
                bi_results["questions"],
            )
            bi_results.update({"accuracy": bi_accuracy, "correct": bi_correct, "total": bi_total})
            print(f"  BlockInfer accuracy: {bi_correct}/{bi_total} = {bi_accuracy:.1f}%")

        if not args.skip_vanilla:
            vl_results = benchmark_vanilla_sdar(
                model_path,
                questions,
                args.max_tokens,
            tokenizer,
            steps=args.denoising_steps,
            block_length=args.block_length,
            temperature=args.temperature,
            top_k=0,
            top_p=1.0,
            remasking_strategy="entropy_bounded",
            confidence_threshold=0.85,
            eb_threshold=args.eb_threshold,
        )
            vl_accuracy, vl_correct, vl_total, vl_details = calculate_accuracy(
                vl_results["outputs"],
                vl_results["questions"],
            )
            vl_results.update({"accuracy": vl_accuracy, "correct": vl_correct, "total": vl_total})
            print(f"  Vanilla accuracy:   {vl_correct}/{vl_total} = {vl_accuracy:.1f}%")

        results_by_source[src] = {"blockinfer_sdar": bi_results, "vanilla_sdar": vl_results}

        if bi_results and vl_results:
            print(f"  Accuracy diff: {bi_results['accuracy'] - vl_results['accuracy']:+.1f}%")
            print(f"  Throughput: BI {bi_results['throughput']:.2f} tok/s vs Vanilla {vl_results['throughput']:.2f} tok/s")

    save_results_multi(results_by_source, args.output)
    
    print("\n" + "="*80)
    print("Benchmark completed!")
    print("="*80)


if __name__ == "__main__":
    main()
