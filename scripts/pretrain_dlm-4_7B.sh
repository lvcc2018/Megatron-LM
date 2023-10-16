#! /bin/bash

# Setting the environment variables
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export MASTER_ADDR=36.103.218.134
export MASTER_PORT=6003

# Distributed training variables
GPUS_PER_NODE=8

# Parallelism variables
TP=1
PP=1

# Network size variables
MODEL_SIZE=7

HIDDEN_SIZE=4096
NUM_HEAD=32
NUM_LAYERS=32
FFN_HIDDEN_SIZE=11008

DROP_OUT=0
MAX_LR=3e-4
MIN_LR=3e-5
MAX_SEQ_LEN=4096

NAME=test_dlm-4-7B

DATA_PATH=/mnt/public/data/DLM-2/65B/word_embedding_97G/word_embedding_97G_text_document

LOG_NAME=llama_${MODEL_SIZE}b_TP${TP}_PP${PP}_MAXLR${MAX_LR}_DROP${DROP_OUT}_${NAME}

SRC_PATH=/mnt/user/lvchuancheng/Megatron-LM/pretrain_gpt.py
LOG_PATH=/mnt/user/lvchuancheng/Megatron-LM/logs/${LOG_NAME}_${RANK}.log
TOKENIZER_PATH=/mnt/public/checkpoint/tokenizer/SentencePieceTokenizer/chinese_llama.model

# wandb environment
export WANDB_API_KEY="e0b30216258c751235154d145c5deab25d92f7b3"
export WANDB_PROJECT="Pretrain"  # project name
export WANDB_ENTITY="deeplang-ai"   # orgnization name
export WANDB_NAME="${LOG_NAME}"  # this run name
export WANDB_NOTES="DLM-4 7B test run"

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes 12 \
       --node_rank ${1} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       "

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --no-position-embedding \
       --max-position-embeddings 4096 \
       --use-rotary-position-embeddings \
       --make-vocab-size-divisible-by 1 \
       --layernorm-epsilon ${NORM_EPS} \
       --normalization rmsnorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --sequence-parallel \
       "

LOGGING_ARGS=" \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
       --micro-batch-size 1 \
       --global-batch-size 2112 \
       --train-iters 5000 \
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
       --lr-warmup-iters 500 \
       --min-lr ${MIN_LR} \
       "

CHECKPOINTING_ARGS=" \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --no-query-key-layer-scaling \
       "

VALIDATION_ARGS=" \
       --eval-interval 2000000 \
       --eval-iters 10 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 1000,0,0 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 0 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
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
       ${VALIDATION_ARGS} \
       ${DATA_ARGS}
       "

${CMD} 2>&1 | tee ${LOG_PATH}