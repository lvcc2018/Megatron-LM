#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO

# Distributed training variables
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Parallelism variables
TP=2
PP=2
DP=$((${WORLD_SIZE}/${TP}/${PP}))

# Paths and names
SRC_PATH=./pretrain_gpt.py
DATA_PATH=./data/meg-gpt2-oscar-en-10k_text_document
SAVE_PATH=./checkpoint
LOG_PATH=./log/node${NODE_RANK}.log
mkdir -p ./log

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       "

NETWORK_SIZE_ARGS=" \
       --num-layers 12 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --max-position-embeddings 1024 \
       --normalization rmsnorm \
       --use-distributed-optimizer \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --position-embedding-type rope \
       "

REGULARIZATION_ARGS=" \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       "

TRAINING_ARGS=" \
       --micro-batch-size 8 \
       --global-batch-size 1024 \
       --train-iters 500000 \
       --log-interval 1 \
       --exit-interval 20 \
       "

LEARNING_RATE_ARGS=" \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --lr-warmup-fraction .01 \
       --min-lr 1.0e-5 \
       "

CHECKPOINTING_ARGS=" \
       --save ${SAVE_PATH} \
       --save-interval 20 \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       "

VALIDATION_ARGS=" \
       --eval-interval 10 \
       --eval-iters 10 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 949,50,1 \
       --seq-length 1024 \
       --vocab-file ./data/gpt2-vocab.json \
       --merge-file ./data/gpt2-merges.txt \
       --data-impl mmap \
       "

export CMD=" \
       ${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${REGULARIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       "

echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}
