#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用 AI 环境完整测试脚本 test_env.py
自动检测 Python / Torch / CUDA / FlashAttention / CXX11 ABI 等信息
最后输出基于检测结果的精炼总结
"""

import os
import sys
import subprocess
import platform
import traceback

summary = []  # 收集总结信息


def ok(msg):
    summary.append("✔ " + msg)
    print("✔", msg)


def warn(msg):
    summary.append("⚠ " + msg)
    print("⚠", msg)


def fail(msg):
    summary.append("❌ " + msg)
    print("❌", msg)


print("=" * 60)
print("🔎 Python 环境信息")
print("=" * 60)

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print("Platform:", platform.platform())
ok("Python 运行正常")

print("\n" + "=" * 60)
print("🔎 系统 CUDA 与驱动检测")
print("=" * 60)


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
    except Exception:
        return ""


nvidia_smi = run_cmd("nvidia-smi")
print("nvidia-smi:")
print(nvidia_smi)
if "Driver Version" in nvidia_smi:
    ok("检测到 NVIDIA GPU 驱动")
else:
    warn("未检测到有效的 NVIDIA 驱动")

nvcc = run_cmd("nvcc --version")
print("nvcc version:")
print(nvcc)
if "release" in nvcc:
    ok("检测到 nvcc 编译器")
else:
    warn("未检测到 nvcc（CUDA Toolkit 可能未安装）")

print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))


print("\n" + "=" * 60)
print("🔎 PyTorch 测试")
print("=" * 60)

torch_available = False
cuda_available = False
abi_flag = None

try:
    import torch
    torch_available = True
    print("PyTorch version:", torch.__version__)
    print("PyTorch built CUDA:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())

    ok("PyTorch 导入成功")

    if torch.cuda.is_available():
        cuda_available = True
        ok("PyTorch CUDA 可用")
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Capability:", torch.cuda.get_device_capability(0))
    else:
        warn("PyTorch 未检测到 GPU")

    print("\nPyTorch Config:")
    config = torch.__config__.show()
    abi_flag = "1" if "TORCH_CXX11_ABI: 1" in config else "0"
    print(config)
    ok(f"TORCH_CXX11_ABI = {abi_flag}")

except Exception:
    fail("PyTorch 测试失败")
    traceback.print_exc()


print("\n" + "=" * 60)
print("🔎 FlashAttention 测试")
print("=" * 60)

flash_ok = False
flash_kernel_ok = False

try:
    import flash_attn
    flash_ok = True
    ok(f"FlashAttention 导入成功: {flash_attn.__version__}")

    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        ok("FlashAttention CUDA 扩展已加载")

        # FlashAttention kernel test (correct shape)
        if cuda_available:
            try:
                B, S, H, D = 1, 64, 8, 64
                q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
                k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
                v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

                out = flash_attn_func(q, k, v)
                flash_kernel_ok = True
                ok("FlashAttention kernel 测试成功")
            except Exception:
                fail("FlashAttention kernel 执行失败（可能是版本/ABI/形状问题）")
                traceback.print_exc()
        else:
            warn("CUDA 不可用，跳过 FlashAttention kernel 测试")

    except Exception:
        fail("FlashAttention CUDA 扩展加载或运行失败")
        traceback.print_exc()

except Exception:
    fail("无法导入 FlashAttention")
    traceback.print_exc()


print("\n" + "=" * 60)
print("🔎 总结")
print("=" * 60)

# 精炼结论逻辑
if not torch_available:
    print("❌ PyTorch 异常：无法运行 AI 相关任务")
elif torch_available and not cuda_available:
    print("⚠ PyTorch 已安装，但未检测到 GPU（可能是驱动或 CUDA 配置问题）")
elif torch_available and cuda_available and abi_flag == "0":
    print("⚠ GPU 可用，但 PyTorch 使用旧 ABI（可能导致扩展库不兼容）")
elif flash_ok and not flash_kernel_ok:
    print("⚠ FlashAttention 加载成功但 kernel 失败（多为 CUDA/PT ABI 版本不一致）")
elif flash_ok and flash_kernel_ok:
    print("✔ 系统已成功加载 PyTorch + CUDA + FlashAttention（环境正常）")
else:
    print("⚠ 部分模块可用，但未完全通过测试")

print("\n详细状态：")
for s in summary:
    print(s)
