#!/bin/bash

export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# This example will start serving the 345M model that is partitioned 8 way tensor parallel
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

HIDDEN_SIZE=4096
NUM_HEAD=32
NUM_LAYERS=32
FFN_HIDDEN_SIZE=11008

CHECKPOINT=<Path to checkpoint>

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32  \
       --hidden-size 4096  \
       --load ${CHECKPOINT}  \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model <Path to tokenizer model> \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 2048  \
       --out-seq-length 2048  \
       --temperature 1.0  \
       --top_p 0.9  \
       --seed 42 \
       --use-distributed-optimizer \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --use-rotary-position-embeddings \
       --no-position-embedding \
       --make-vocab-size-divisible-by 1 \
       --layernorm-epsilon 1e-6 \
       --normalization rmsnorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --no-merge-qkv \
       --disable-bias-linear
