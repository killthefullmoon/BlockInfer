#!/bin/bash

# Setup environment for BlockInfer
export PATH=/home/hymanzzs/.conda/envs/dllm-rl/bin:$PATH
export CUDA_HOME=/usr/local/cuda

# Run LLaDA with BlockInfer engine
/home/hymanzzs/.conda/envs/dllm-rl/bin/python example.py

