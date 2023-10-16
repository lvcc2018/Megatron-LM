#!/bin/bash

python tools/convert_checkpoint/merge_and_split_pp_model.py \
    --input-model-dir /mnt/data/lvchuancheng/checkpoint/LLaMA-2/llama-2-70b-mgt-TP4-PP4-qkv/release \
    --input-tp 4 \
    --input-pp 4 \
    --target-model-dir /mnt/data/lvchuancheng/checkpoint/LLaMA-2/llama-2-70b-mgt-TP8-PP1-qkv \
    --target-tp 8 \
    --target-pp 1 \
    --target-dp 1 \
    --num-layers 80 \
    --merge_qkv
