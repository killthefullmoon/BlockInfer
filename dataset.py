from datasets import load_dataset, Dataset, concatenate_datasets
import random
import re
import json

def sample(ds, n):
    if len(ds) <= n:
        return ds
    idx = random.sample(range(len(ds)), n)
    return ds.select(idx)

def parse_gsm8k_answer(a_raw: str):
    m = re.search(r"####\s*([^\n]+)", a_raw)
    if m:
        return m.group(1).strip()
    return a_raw.strip()

def extract_fields(example, source_name):
    q_keys = ["question", "problem", "input", "query", "prompt"]
    a_keys = ["answer", "answers", "solution", "label", "target"]
    c_keys = ["choices", "options"]

    q = None
    a = None
    choices = None

    # Question
    for k in q_keys:
        if k in example:
            q = example[k]
            break

    # Answer
    for k in a_keys:
        if k in example:
            a = example[k]
            break

    # Choices
    for k in c_keys:
        if k in example:
            choices = example[k]
            break

    # ------ Normalize answer ------
    if isinstance(a, list):
        a = a[0] if len(a) else ""

    if a is None:
        a = ""

    # GSM8K special rule
    if source_name == "gsm8k":
        a = parse_gsm8k_answer(a)

    # ------ Normalize choices ------
    # HF concat cannot mix Sequence(string) and null
    # → Convert ALL choices to a JSON string
    if choices is None:
        choices = ""
    else:
        if isinstance(choices, dict):
            choices = list(choices.values())
        # convert to json string
        choices = json.dumps(choices)

    return {
        "question": str(q),
        "answer": str(a),
        "choices": str(choices),  # always string
        "source": source_name
    }


def to_uniform(ds, source_name):
    return Dataset.from_list([extract_fields(x, source_name) for x in ds])


def main():
    total = 200
    per = total // 4

    print("Loading GSM8K...")
    gsm8k = load_dataset("gsm8k", "main", split="test")
    gsm8k = to_uniform(sample(gsm8k, per), "gsm8k")

    print("Loading Math500...")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math500 = to_uniform(sample(math500, per), "math500")

    print("Loading LongBench (hotpotqa)...")
    lb = load_dataset("THUDM/LongBench", "hotpotqa", split="test", trust_remote_code=True)
    lb = to_uniform(sample(lb, per), "longbench_hotpotqa")

    print("Loading MMLU (cais/mmlu)...")
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu = to_uniform(sample(mmlu, per), "mmlu")

    print("Merging...")
    final = concatenate_datasets([gsm8k, math500, lb, mmlu])

    print("Final size:", len(final))
    final.to_json("sample_200.jsonl")
    print("Saved to sample_200.jsonl")


if __name__ == "__main__":
    main()
