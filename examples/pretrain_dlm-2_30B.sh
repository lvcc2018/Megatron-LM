#! /bin/bash

# Continue training the LLaMA model
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

# Distributed training variables
GPUS_PER_NODE=8
NNODES=20
NODE_RANK=${1}
MASTER_ADDR="192.168.48.2"
MASTER_PORT=6002
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))

# Parallelism variables
TP=4
PP=2
DP=$((${WORLD_SIZE}/${TP}/${PP}))

# Network size variables
MODEL_SIZE=30

if   [[ ${MODEL_SIZE} == 7 ]];    then HIDDEN_SIZE=4096;  NUM_HEAD=32;  NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008
elif [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=40;  NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=6656;  NUM_HEAD=52;  NUM_LAYERS=60; FFN_HIDDEN_SIZE=17920
elif [[ ${MODEL_SIZE} == 65 ]];   then HIDDEN_SIZE=8192;  NUM_HEAD=64;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=22016
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


DROP_OUT=0.1
MAX_LR=5e-5
MIN_LR=5e-6
NAME=zh60_en40_code50_dynamic_maxlen4096

DATA_PATH="0.1657-0.4419-0.2914 zh /mnt/public/data/DLM-2/30B/pretrain/zh60_text_document"          # 60949945058
DATA_PATH+=" 0.4972-0.2210-0.2914 en /mnt/public/data/DLM-2/30B/pretrain/en40_text_document"        # 39635205055
DATA_PATH+=" 0.3371 code /mnt/public/data/DLM-2/30B/pretrain/code50_text_document"                # 51147486813

LOG_NAME=llama_${MODEL_SIZE}b_WS${WORLD_SIZE}_TP${TP}_PP${PP}_MAXLR${MAX_LR}_DROP${DROP_OUT}_${NAME}

SRC_PATH=/mnt/lvchuancheng/Megatron-LM/pretrain_gpt.py
SAVE_PATH=/mnt/checkpoint/${LOG_NAME}
LOAD_PATH=/mnt/public/checkpoint/megatron_llama/30B/origin_TP4_PP2_DP20_warmup
LOG_PATH=/mnt/public/log/${LOG_NAME}_${NODE_RANK}.log

# wandb environment
export WANDB_API_KEY="e0b30216258c751235154d145c5deab25d92f7b3"
export WANDB_PROJECT="Pretrain"  # project name
export WANDB_ENTITY="deeplang-ai"   # orgnization name
export WANDB_NAME="${LOG_NAME}"  # this run name
export WANDB_NOTES="DLM-2-30B pretraining" # Longer notes about your run. Markdown is allowed and you can edit this later in the UI.

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
       --use-distributed-optimizer \
       "

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --max-position-embeddings 4096 \
       --use-rotary-position-embeddings \
       --no-position-embedding \
       --make-vocab-size-divisible-by 4 \
       --layernorm-epsilon 1e-6 \
       --normalization rmsnorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --no-merge-qkv \
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
       --global-batch-size 1600 \
       --train-iters 23000 \
       --log-interval 1 \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --use-flash-attn \
       --optimizer adam \
       "

INITIALIZATION_ARGS=" \
       --seed 42 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-iters 2300 \
       --min-lr ${MIN_LR} \
       "

CHECKPOINTING_ARGS=" \
       --save ${SAVE_PATH} \
       --save-interval 200 \
       --load ${LOAD_PATH} \
       --no-load-optim \
       --no-load-rng \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --no-query-key-layer-scaling \
       "

VALIDATION_ARGS=" \
       --eval-interval 200 \
       --eval-iters 10 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 998,1,1 \
       --seq-length 4096 \
       --num-workers 0 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model /mnt/public/checkpoint/tokenizer/SentencePieceTokenizer/chinese_llama.model \
       --data-impl mmap \
       --dataloader-type single \
       --use-dataset-manager \
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