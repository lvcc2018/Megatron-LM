#! /bin/bash

# Setting the environment variables
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
export MASTER_ADDR=172.16.11.11
export MASTER_PORT=6003
export GLOO_SOCKET_IFNAME=bond4
export NCCL_SOCKET_IFNAME=bond4

# Distributed training variables
GPUS_PER_NODE=8
RANK=${1}

# Parallelism variables
TP=1
PP=1

# Network size variables
MODEL_SIZE=1.3

HIDDEN_SIZE=2048
NUM_HEAD=32
NUM_LAYERS=24
FFN_HIDDEN_SIZE=5632

DROP_OUT=0.1
MAX_LR=5e-4
MIN_LR=5e-5
MAX_SEQ_LEN=4096

NAME=pretrain_dlms_1.3B_en100_zh100

DATA_PATH="0.5000 zh /mnt/public/data/DLM-4/zh/bin/zh_text_document"
DATA_PATH+=" 0.5000 en /mnt/public/data/DLM-4/en/bin/en_text_document"

LOG_NAME=${NAME}_${MODEL_SIZE}b_TP${TP}_PP${PP}_MAXLR${MAX_LR}_DROP${DROP_OUT}

SRC_PATH=/mnt/user/lvchuancheng/Megatron-LM/pretrain_dlm.py
LOG_PATH=/mnt/user/lvchuancheng/Megatron-LM/logs/${LOG_NAME}_${RANK}.log
SAVE_PATH=/mnt/user/lvchuancheng/Megatron-LM/checkpoint/${LOG_NAME}
LOAD_PATH=/mnt/user/lvchuancheng/Megatron-LM/checkpoint/pretrain_dlms_1.3B_en100_zh100_1.3b_TP1_PP1_MAXLR5e-4_DROP0.1
TOKENIZER_PATH=/mnt/public/checkpoint/tokenizer/SentencePieceTokenizer/chinese_llama.model

# wandb environment
export WANDB_API_KEY="e0b30216258c751235154d145c5deab25d92f7b3"
export WANDB_PROJECT="DLMS-Pretrain"  # project name
export WANDB_ENTITY="deeplang-ai"   # orgnization name
export WANDB_NAME="${LOG_NAME}"  # this run name
export WANDB_NOTES="DLM-S 1.3B test run"

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes 8 \
       --node_rank ${RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       "

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --no-position-embedding \
       --max-position-embeddings ${MAX_SEQ_LEN} \
       --position-embedding-type rope \
       --make-vocab-size-divisible-by 1 \
       --layernorm-epsilon 1e-5 \
       --normalization RMSNorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       "

LOGGING_ARGS=" \
       --log-timers-to-tensorboard \
       --log-memory-to-tensorboard \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-5 \
       "

TRAINING_ARGS=" \
       --micro-batch-size 4 \
       --global-batch-size 2048 \
       --train-iters 25000 \
       --log-interval 1 \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --use-flash-attn \
       --optimizer adam \
       "

INITIALIZATION_ARGS=" \
       --seed 1403 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-iters 2000 \
       --min-lr ${MIN_LR} \
       --distributed-timeout-minutes 30 \
       "

CHECKPOINTING_ARGS=" \
       --save ${SAVE_PATH} \
       --save-interval 2000 \
       --load ${LOAD_PATH} \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --no-query-key-layer-scaling \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 1000,0,0 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 0 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --use-dataset-manager \
       --data-impl mmap \
       "
       
export CMD="${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${DATA_ARGS}
       "

${CMD} 2>&1 | tee ${LOG_PATH}