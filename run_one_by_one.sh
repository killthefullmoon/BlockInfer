#!/bin/bash
# Run benchmark one sample at a time in terminal

cd /nfs/turbo/coe-zmao/hymanzzs/BlockInfer

# Add Fast-dLLM to path
export PYTHONPATH=/nfs/turbo/coe-zmao/hymanzzs/Fast-dLLM:$PYTHONPATH

# Configuration - modify these as needed
SOURCE=""  # "gsm8k", "math500", "mmlu", "longbench_hotpotqa" or empty for all
MAX_TOKENS=256
SKIP_VANILLA=""  # "--skip-vanilla" to skip vanilla

# Count total samples
if [ -z "$SOURCE" ]; then
    TOTAL=$(wc -l < sample_200.jsonl)
else
    TOTAL=$(grep "\"source\": \"$SOURCE\"" sample_200.jsonl | wc -l)
fi

echo "======================================"
echo "LLaDA Benchmark - One by One"
echo "Total: $TOTAL samples"
echo "======================================"

# Process each sample
for i in $(seq 100 $((TOTAL-1))); do
    echo ""
    echo "[Sample $((i+1))/$TOTAL]"
    
    CMD="python benchmark_llada.py --num-samples 1 --skip-sample $i --max-tokens $MAX_TOKENS --output benchmark_sample_${i}.json"
    [ -n "$SOURCE" ] && CMD="$CMD --source $SOURCE"
    [ -n "$SKIP_VANILLA" ] && CMD="$CMD $SKIP_VANILLA"
    
    eval $CMD
    
    [ $? -eq 0 ] && echo "✓ Done" || echo "✗ Failed"
done

echo ""
echo "======================================"
echo "Aggregating results..."
echo "======================================"

# Aggregate
python -c "
import json, glob

bi_times, bi_tokens, vl_times, vl_tokens = [], [], [], []

for f in sorted(glob.glob('benchmark_sample_*.json')):
    try:
        with open(f) as fp:
            d = json.load(fp)
        if 'blockinfer' in d:
            bi_times.append(d['blockinfer'].get('total_time', 0))
            bi_tokens.append(d['blockinfer'].get('total_tokens', 0))
        if 'vanilla_llada' in d:
            vl_times.append(d['vanilla_llada'].get('total_time', 0))
            vl_tokens.append(d['vanilla_llada'].get('total_tokens', 0))
    except: pass

print('\\nFINAL RESULTS')
print('='*40)
if bi_times:
    t, tok = sum(bi_times), sum(bi_tokens)
    print(f'BlockInfer: {t:.2f}s, {tok} tokens, {tok/t:.2f} tok/s')
if vl_times:
    t, tok = sum(vl_times), sum(vl_tokens)
    print(f'Vanilla:    {t:.2f}s, {tok} tokens, {tok/t:.2f} tok/s')

json.dump({'blockinfer': {'times': bi_times, 'tokens': bi_tokens}, 'vanilla': {'times': vl_times, 'tokens': vl_tokens}}, open('benchmark_final_results.json', 'w'), indent=2)
print('\\nSaved to benchmark_final_results.json')
"



