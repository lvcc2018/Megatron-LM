#! /bin/bash

# Continue training the LLaMA model
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Distributed training variables
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))

# Parallelism variables
TP=2
PP=2
DP=$((${WORLD_SIZE}/${TP}/${PP}))

# Network size variables
MODEL_SIZE=7

if   [[ ${MODEL_SIZE} == 7 ]];    then HIDDEN_SIZE=4096;  NUM_HEAD=32;  NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008
elif [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=40;  NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=6656;  NUM_HEAD=52;  NUM_LAYERS=60; FFN_HIDDEN_SIZE=17920
elif [[ ${MODEL_SIZE} == 65 ]];   then HIDDEN_SIZE=8192;  NUM_HEAD=64;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=22016
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

# To test variables
DROP_OUT=0
MAX_LR=1e-5
MIN_LR=1e-6


# Paths and names
LOG_NAME=llama_${MODEL_SIZE}b_WS${WORLD_SIZE}_TP${TP}_PP${PP}_MAXLR${MAX_LR}_DROP${DROP_OUT}

STORAGE_PATH=/mnt/data01/huangyufei
MEGATRON_PATH=${STORAGE_PATH}/Megatron-LM
DATA_PATH="1 en /mnt/data01/shenyan/data/DLM-2-data/7B/all/en/dlm-2-7B-en_text_document"
DATA_PATH+=" 2 zh /mnt/data01/shenyan/data/DLM-2-data/7B/all/zh/dlm-2-7B-zh_text_document"
SRC_PATH=${MEGATRON_PATH}/pretrain_gpt.py
SAVE_PATH=${MEGATRON_PATH}/checkpoint/${LOG_NAME}
# LOAD_PATH=${STORAGE_PATH}/pretrained_models/llama_chinese/llama_7b_WS16_TP1_PP1
LOAD_PATH=${MEGATRON_PATH}/checkpoint/llama_7b_WS8_TP2_PP2_MAXLR1e-5_DROP0
LOG_PATH=${MEGATRON_PATH}/log/${LOG_NAME}.log
TSBD_PATH=${MEGATRON_PATH}/tensorboard/${LOG_NAME}
SAVE_INTERVAL=1000
EVAL_INTERVAL=5
TRAIN_ITERS=4000

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
       --max-position-embeddings 2048 \
       --use-rotary-position-embeddings \
       --no-position-embedding \
       --make-vocab-size-divisible-by 1 \
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
       --global-batch-size 128 \
       --train-iters ${TRAIN_ITERS} \
       --log-interval 1 \
       --tensorboard-dir ${TSBD_PATH}
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --use-flash-attn \
       --optimizer adam \
       --exit-interval 40000 \
       "

INITIALIZATION_ARGS=" \
       --seed 1403 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-iters 500 \
       --lr-decay-iters 1500 \
       --min-lr ${MIN_LR} \
       "

CHECKPOINTING_ARGS=" \
       --save ${SAVE_PATH} \
       --save-interval ${SAVE_INTERVAL} \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --no-query-key-layer-scaling \
       "

VALIDATION_ARGS=" \
       --eval-interval ${EVAL_INTERVAL} \
       --eval-iters 20 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 998,1,1 \
       --seq-length 2048 \
       --num-workers 0 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model /mnt/data01/shenyan/lvcc/DLM-2/Exp_7B/tokenizer/SentencePieceTokenizer/chinese_llama.model \
       --data-impl mmap \
       --use-dataloader-manager \
       --dataloader-type cyclic \
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