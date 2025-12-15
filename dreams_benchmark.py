import argparse
import json
import re

from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer


def find_cached_model(model_id: str) -> str | None:
    """
    Return the latest local HF cache snapshot path for the given model_id, if it exists.
    This mirrors the helper from the dreams repo so the script works standalone.
    """
    import os
    from pathlib import Path

    cache_dir = None
    try:
        from huggingface_hub import file_download
        cache_dir = getattr(file_download, "HUGGINGFACE_HUB_CACHE", None)
    except Exception:
        cache_dir = None

    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = os.path.join(hf_home, "hub")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    snap_root = Path(cache_dir) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snap_root.is_dir():
        return None

    snapshots = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in snapshots:
        if (snap / "config.json").is_file():
            return str(snap)
    return None


def load_questions(path, source=None, limit=None):
    """Load questions from JSONL, optionally filtering by source."""
    items = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            if source and data.get("source") != source:
                continue
            items.append(data)
            if limit and len(items) >= limit:
                break
    return items


def format_prompt(question, source):
    """Apply a simple task hint tuned for math500 style problems."""
    if source == "math500":
        return f"{question}\n\nSolve this problem and put your final answer in curly braces like {{answer}}."
    return question


def extract_answer(text, source):
    """Extract answer from generated text (mirrors benchmark_dream logic)."""
    brace_match = re.search(r"\{([^}]+)\}", text)
    if brace_match:
        answer = brace_match.group(1).strip()
        if source in ["gsm8k", "math500"]:
            answer = answer.replace(",", "").strip()
        return answer

    if source == "gsm8k":
        patterns = [
            r"####\s*([0-9,\.]+)",
            r"[Aa]nswer:\s*([0-9,\.]+)",
            r"[Rr]esult:\s*([0-9,\.]+)",
            r"=\s*([0-9,\.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(",", "")
        numbers = re.findall(r"[0-9,\.]+", text)
        if numbers:
            return numbers[-1].replace(",", "")

    if source == "math500":
        patterns = [
            r"\\boxed\{([^}]+)\}",
            r"[Aa]nswer:\s*(.+?)(?:\.|<|$)",
            r"[Ff]inal answer:\s*(.+?)(?:\.|<|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    if source == "mmlu":
        match = re.search(r"(?:^|[^\d])([0-3])(?:[^\d]|$)", text)
        if match:
            return match.group(1)
        match = re.search(r"[0-3]", text)
        if match:
            return match.group(0)

    if source == "longbench_hotpotqa":
        patterns = [
            r"[Aa]nswer:\s*(.+?)(?:\.|<|$)",
            r"[Tt]he answer is\s*(.+?)(?:\.|<|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    words = text.split()[:10]
    return " ".join(words) if words else text[:100]


def normalize_answer(ans):
    """Normalize answers for loose string matching."""
    ans = str(ans)
    return ans.lower().replace(" ", "").replace(",", "")


def main():
    parser = argparse.ArgumentParser(description="Run Dream diffusion_generate over a JSONL dataset.")
    parser.add_argument("--dataset", type=str, default="sample_200.jsonl", help="Path to JSONL dataset.")
    parser.add_argument("--source", type=str, default="math500", help="Filter by source key in JSONL.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to run.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--steps", type=int, default=32, help="Diffusion steps.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus value.")
    args = parser.parse_args()

    questions = load_questions(args.dataset, args.source, args.max_samples)
    if not questions:
        raise ValueError(f"No questions found in {args.dataset} with source={args.source}")

    model_id = "Dream-org/Dream-v0-Instruct-7B"
    cached_path = find_cached_model(model_id)
    model_path = cached_path or model_id
    print(f"Using model path: {model_path}")

    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    correct = 0

    for idx, item in enumerate(tqdm(questions, desc="Running", unit="q"), start=1):
        prompt = format_prompt(item.get("question", ""), args.source)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device=device)
        attention_mask = inputs.attention_mask.to(device=device)

        with torch.inference_mode():
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg="entropy",
                alg_temp=0.0,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        sequences = output.sequences if hasattr(output, "sequences") else output
        generations = [
            tokenizer.decode(seq[len(p) :].tolist())
            for p, seq in zip(input_ids, sequences)
        ]

        gold = item.get("answer", "")
        pred = extract_answer(generations[0], args.source)
        gold_norm = normalize_answer(gold)
        pred_norm = normalize_answer(pred)
        is_correct = gold_norm == pred_norm if gold_norm and pred_norm else False
        correct += int(is_correct)

        print(f"\n[{idx}/{len(questions)}] Prompt:")
        print(prompt)
        print("\nGeneration:")
        print(generations[0].split(tokenizer.eos_token)[0])
        if gold:
            print(f"Gold answer: {gold}")
            print(f"Pred answer: {pred}")
            print(f"Match: {'YES' if is_correct else 'NO'}")

    accuracy = correct / len(questions) * 100 if questions else 0.0
    print("\nFinished inference.")
    print(f"Total prompts: {len(questions)}")
    print(f"Accuracy: {correct}/{len(questions)} = {accuracy:.1f}%")


if __name__ == "__main__":
    main()
