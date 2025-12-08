#!/usr/bin/env python3
"""
分析 BlockInfer 和 Vanilla_llada 的 benchmark 结果
按数据集（gsm8k, math500, longbench_hotpotqa, mmlu）分别统计
对比两种方法的正确率、throughput 和 throughput_speedup
"""

import json
import os
import re
from pathlib import Path
import numpy as np

def load_samples(sample_file):
    """加载 sample_200.jsonl 获取每个样本的数据集来源"""
    samples = []
    with open(sample_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                samples.append(data)
    return samples

def load_results(folder_path):
    """加载指定文件夹中的所有 benchmark 结果，并按样本索引排序"""
    results = {}
    folder = Path(folder_path)
    
    for json_file in folder.glob("benchmark_sample_*.json"):
        # 提取样本索引 (e.g., benchmark_sample_4.json -> 4, benchmark_sample_4_128.json -> 4)
        filename = json_file.stem
        match = re.search(r'benchmark_sample_(\d+)', filename)
        if match:
            idx = int(match.group(1))
            with open(json_file, 'r') as f:
                data = json.load(f)
                results[idx] = data
    
    return results

def analyze_by_dataset(results, samples, name):
    """按数据集分析结果"""
    # 按数据集分组
    datasets = {}
    for idx, result in results.items():
        if idx < len(samples):
            source = samples[idx].get('source', 'unknown')
            if source not in datasets:
                datasets[source] = []
            datasets[source].append((idx, result))
    
    print(f"\n{'='*70}")
    print(f"📊 {name} 结果分析 (共 {len(results)} 个样本)")
    print(f"{'='*70}")
    
    all_stats = {}
    
    # 按数据集顺序排序
    dataset_order = ['gsm8k', 'math500', 'longbench_hotpotqa', 'mmlu']
    sorted_datasets = sorted(datasets.keys(), key=lambda x: dataset_order.index(x) if x in dataset_order else 999)
    
    for source in sorted_datasets:
        data_list = datasets[source]
        
        blockinfer_correct = 0
        blockinfer_total = 0
        vanilla_correct = 0
        vanilla_total = 0
        blockinfer_throughputs = []
        vanilla_throughputs = []
        throughput_speedups = []
        
        for idx, r in data_list:
            blockinfer_correct += r['blockinfer']['correct']
            blockinfer_total += r['blockinfer']['total']
            vanilla_correct += r['vanilla_llada']['correct']
            vanilla_total += r['vanilla_llada']['total']
            blockinfer_throughputs.append(r['blockinfer']['throughput'])
            vanilla_throughputs.append(r['vanilla_llada']['throughput'])
            throughput_speedups.append(r['comparison']['throughput_speedup'])
        
        bi_acc = blockinfer_correct / blockinfer_total if blockinfer_total > 0 else 0
        va_acc = vanilla_correct / vanilla_total if vanilla_total > 0 else 0
        
        stats = {
            'source': source,
            'num_samples': len(data_list),
            'blockinfer_correct': blockinfer_correct,
            'blockinfer_total': blockinfer_total,
            'blockinfer_accuracy': bi_acc,
            'vanilla_correct': vanilla_correct,
            'vanilla_total': vanilla_total,
            'vanilla_accuracy': va_acc,
            'accuracy_diff': bi_acc - va_acc,
            'blockinfer_throughput_mean': np.mean(blockinfer_throughputs),
            'blockinfer_throughput_std': np.std(blockinfer_throughputs),
            'vanilla_throughput_mean': np.mean(vanilla_throughputs),
            'vanilla_throughput_std': np.std(vanilla_throughputs),
            'speedup_mean': np.mean(throughput_speedups),
            'speedup_std': np.std(throughput_speedups),
            'speedup_min': np.min(throughput_speedups),
            'speedup_max': np.max(throughput_speedups),
        }
        all_stats[source] = stats
    
    # 打印每个数据集的结果
    print(f"\n📌 各数据集正确率 (Accuracy)")
    print(f"   {'数据集':<25} {'样本数':<8} {'BI正确':<8} {'VA正确':<8} {'BI准确率':<12} {'VA准确率':<12} {'差异':<10}")
    print(f"   {'-'*93}")
    for source in sorted_datasets:
        s = all_stats[source]
        print(f"   {source:<25} {s['num_samples']:<8} {s['blockinfer_correct']:<8} {s['vanilla_correct']:<8} {s['blockinfer_accuracy']*100:<12.2f} {s['vanilla_accuracy']*100:<12.2f} {s['accuracy_diff']*100:+.2f}%")
    
    # 计算总体统计
    total_bi_correct = sum(s['blockinfer_correct'] for s in all_stats.values())
    total_bi_total = sum(s['blockinfer_total'] for s in all_stats.values())
    total_va_correct = sum(s['vanilla_correct'] for s in all_stats.values())
    total_va_total = sum(s['vanilla_total'] for s in all_stats.values())
    total_bi_acc = total_bi_correct / total_bi_total if total_bi_total > 0 else 0
    total_va_acc = total_va_correct / total_va_total if total_va_total > 0 else 0
    print(f"   {'-'*93}")
    print(f"   {'总计':<25} {len(results):<8} {total_bi_correct:<8} {total_va_correct:<8} {total_bi_acc*100:<12.2f} {total_va_acc*100:<12.2f} {(total_bi_acc-total_va_acc)*100:+.2f}%")
    
    print(f"\n📌 各数据集 Throughput (tokens/sec)")
    print(f"   {'数据集':<25} {'BI平均':<12} {'BI标准差':<12} {'VA平均':<12} {'VA标准差':<12}")
    print(f"   {'-'*73}")
    for source in sorted_datasets:
        s = all_stats[source]
        print(f"   {source:<25} {s['blockinfer_throughput_mean']:<12.2f} {s['blockinfer_throughput_std']:<12.2f} {s['vanilla_throughput_mean']:<12.2f} {s['vanilla_throughput_std']:<12.2f}")
    
    print(f"\n📌 各数据集 Throughput Speedup (BlockInfer / Vanilla)")
    print(f"   {'数据集':<25} {'平均加速':<12} {'标准差':<12} {'最小':<12} {'最大':<12}")
    print(f"   {'-'*73}")
    for source in sorted_datasets:
        s = all_stats[source]
        print(f"   {source:<25} {s['speedup_mean']:<12.4f} {s['speedup_std']:<12.4f} {s['speedup_min']:<12.4f} {s['speedup_max']:<12.4f}")
    
    return all_stats

def print_summary(stats_128, stats_256):
    """打印汇总对比"""
    print(f"\n{'='*70}")
    print(f"📊 汇总对比: 128 tokens vs 256 tokens")
    print(f"{'='*70}")
    
    dataset_order = ['gsm8k', 'math500', 'longbench_hotpotqa', 'mmlu']
    
    print(f"\n📌 正确率对比")
    print(f"   {'数据集':<20} {'128t BI':<10} {'128t VA':<10} {'256t BI':<10} {'256t VA':<10}")
    print(f"   {'-'*60}")
    for source in dataset_order:
        if source in stats_128 and source in stats_256:
            s128 = stats_128[source]
            s256 = stats_256[source]
            print(f"   {source:<20} {s128['blockinfer_accuracy']*100:<10.1f} {s128['vanilla_accuracy']*100:<10.1f} {s256['blockinfer_accuracy']*100:<10.1f} {s256['vanilla_accuracy']*100:<10.1f}")
    
    print(f"\n📌 Speedup 对比")
    print(f"   {'数据集':<20} {'128 tokens':<15} {'256 tokens':<15} {'差异':<15}")
    print(f"   {'-'*65}")
    for source in dataset_order:
        if source in stats_128 and source in stats_256:
            s128 = stats_128[source]
            s256 = stats_256[source]
            diff = s256['speedup_mean'] - s128['speedup_mean']
            print(f"   {source:<20} {s128['speedup_mean']:<15.4f} {s256['speedup_mean']:<15.4f} {diff:+.4f}")
    
    # 计算整体平均
    avg_speedup_128 = np.mean([s['speedup_mean'] for s in stats_128.values()])
    avg_speedup_256 = np.mean([s['speedup_mean'] for s in stats_256.values()])
    print(f"   {'-'*65}")
    print(f"   {'整体平均':<20} {avg_speedup_128:<15.4f} {avg_speedup_256:<15.4f} {avg_speedup_256 - avg_speedup_128:+.4f}")

def main():
    base_path = Path(__file__).parent
    
    # 加载样本信息
    print("正在加载样本信息...")
    samples = load_samples(base_path / "sample_200.jsonl")
    print(f"✅ 加载了 {len(samples)} 个样本")
    
    # 统计每个数据集的样本数
    source_counts = {}
    for s in samples:
        source = s.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"   数据集分布: {source_counts}")
    
    # 加载两组结果
    print("\n正在加载 benchmark 结果...")
    results_128 = load_results(base_path / "benchmark_samples_128")
    results_256 = load_results(base_path / "benchmark_samples")
    print(f"✅ 128 tokens: {len(results_128)} 个结果, 256 tokens: {len(results_256)} 个结果")
    
    # 分析各组结果
    stats_128 = analyze_by_dataset(results_128, samples, "128 Tokens")
    stats_256 = analyze_by_dataset(results_256, samples, "256 Tokens")
    
    # 打印汇总对比
    print_summary(stats_128, stats_256)
    
    # 最终总结
    print(f"\n{'='*70}")
    print(f"📊 最终总结")
    print(f"{'='*70}")
    
    # 计算各项整体平均
    total_bi_acc_128 = sum(s['blockinfer_correct'] for s in stats_128.values()) / sum(s['blockinfer_total'] for s in stats_128.values())
    total_va_acc_128 = sum(s['vanilla_correct'] for s in stats_128.values()) / sum(s['vanilla_total'] for s in stats_128.values())
    total_bi_acc_256 = sum(s['blockinfer_correct'] for s in stats_256.values()) / sum(s['blockinfer_total'] for s in stats_256.values())
    total_va_acc_256 = sum(s['vanilla_correct'] for s in stats_256.values()) / sum(s['vanilla_total'] for s in stats_256.values())
    
    avg_speedup_128 = np.mean([s['speedup_mean'] for s in stats_128.values()])
    avg_speedup_256 = np.mean([s['speedup_mean'] for s in stats_256.values()])
    
    print(f"\n🔹 整体正确率:")
    print(f"   128 tokens: BlockInfer {total_bi_acc_128*100:.1f}%, Vanilla {total_va_acc_128*100:.1f}%, 差异 {(total_bi_acc_128-total_va_acc_128)*100:+.1f}%")
    print(f"   256 tokens: BlockInfer {total_bi_acc_256*100:.1f}%, Vanilla {total_va_acc_256*100:.1f}%, 差异 {(total_bi_acc_256-total_va_acc_256)*100:+.1f}%")
    
    print(f"\n🔹 整体加速比:")
    print(f"   128 tokens: {avg_speedup_128:.4f}x")
    print(f"   256 tokens: {avg_speedup_256:.4f}x")
    print(f"   平均: {(avg_speedup_128 + avg_speedup_256) / 2:.4f}x")
    
    print(f"\n🔹 各数据集最佳加速比 (256 tokens):")
    for source in ['gsm8k', 'math500', 'longbench_hotpotqa', 'mmlu']:
        if source in stats_256:
            print(f"   {source}: {stats_256[source]['speedup_mean']:.4f}x")

if __name__ == "__main__":
    main()
