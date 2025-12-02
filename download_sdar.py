#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
下载 HuggingFace 上的 JetLM/SDAR-4B-Chat 模型到本地目录
支持断点续传、自动创建目录、自动选择 fastest mirror
"""

from huggingface_hub import snapshot_download
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Download SDAR-4B-Chat model")
    parser.add_argument(
        "--output",
        type=str,
        default="./models/SDAR-4B-Chat",
        help="下载保存的目标目录"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="模型版本/分支，如 main、v1.0 等"
    )
    args = parser.parse_args()

    repo_id = "JetLM/SDAR-4B-Chat"

    print("📥 开始下载模型:", repo_id)
    print("📁 保存目录:", args.output)
    print("🔄 如中途中断，下次会自动断点续传\n")

    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=args.output,
        local_dir_use_symlinks=False,  # 彻底下载文件
        revision=args.revision,
    )

    print("\n✅  下载完成！")
    print("📌 本地模型目录:", local_path)


if __name__ == "__main__":
    main()
