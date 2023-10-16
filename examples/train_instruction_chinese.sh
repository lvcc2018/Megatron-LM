#! /bin/bash

# Continue training the LLaMA model
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
# MASTER_ADDR="10.1.99.206"
MASTER_PORT=10002
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=8
PP=1
DP=1

MODEL_SIZE=7

if   [[ ${MODEL_SIZE} == 7 ]];    then HIDDEN_SIZE=4096;  NUM_HEAD=32;  NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008
elif [[ ${MODEL_SIZE} == 13 ]];   then HIDDEN_SIZE=5120;  NUM_HEAD=40;  NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824
elif [[ ${MODEL_SIZE} == 30 ]];   then HIDDEN_SIZE=6656;  NUM_HEAD=52;  NUM_LAYERS=60; FFN_HIDDEN_SIZE=17920
elif [[ ${MODEL_SIZE} == 65 ]];   then HIDDEN_SIZE=8192;  NUM_HEAD=64;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=22016
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"
fi

STORAGE_PATH=/storage/huangyufei
BASE_PATH=/storage/huangyufei/Megatron-LM
LOG_NAME=llama_${MODEL_SIZE}b_WS${WORLD_SIZE}_TP${TP}_PP${PP}_instruction_chinese_iter0018500_debug
# DATA_PATH="0.5 en ${BASE_PATH}/datas/first_preprocess/alpaca_data_en_10.json"
# DATA_PATH+=" 0.5 zh ${BASE_PATH}/datas/first_preprocess/alpaca_data_zh_13.json"
DATA_PATH+="0.5 en ${BASE_PATH}/datas/sixth_preprocess/en/coig_en_train.json"
DATA_PATH+=" 0.5 zh ${BASE_PATH}/datas/sixth_preprocess/zh/coig_zh_train.json"
# VALID_DATA_PATH="0.5 en ${BASE_PATH}/datas/first_preprocess/alpaca_data_en_13.json"
# VALID_DATA_PATH+=" 0.5 zh ${BASE_PATH}/datas/first_preprocess/alpaca_data_zh_13.json"
VALID_DATA_PATH+="0.5 en ${BASE_PATH}/datas/sixth_preprocess/en/coig_en_dev.json"
VALID_DATA_PATH+=" 0.5 zh ${BASE_PATH}/datas/sixth_preprocess/zh/coig_zh_dev.json"
# LOAD_PATH="/storage/huangyufei/pretrained_models/llama_chinese/llama_30b_WS64_TP${TP}_PP${PP}"
LOAD_PATH="/storage/huangyufei/pretrained_models/llama_chinese/llama_7b_WS16_TP${TP}_PP${PP}_new"

SRC_PATH=tasks/main.py
# CHECKPOINT_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}_node${NODE_RANK}
CHECKPOINT_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
# LOG_PATH=${BASE_PATH}/log/${LOG_NAME}_node${NODE_RANK}.log
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}.log
TENSORBOARD_PATH=${BASE_PATH}/tensorboard/${LOG_NAME}
DROP_OUT=0.1

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
       --position-embedding-type rotary \
       --make-vocab-size-divisible-by 1 \
       --layernorm-epsilon 1e-6 \
       --normalization rmsnorm \
       --ffn-type SwiGLU \
       --no-tied-lm-head \
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
       --finetune \
       --task instruction \
       --epochs 1 \
       --train-iters 1000 \
       --micro-batch-size 4 \
       --global-batch-size 128 \
       --log-interval 1 \
       --tensorboard-dir ${TENSORBOARD_PATH} \
       --no-bias \
       --no-bias-gelu-fusion \
       --use-flash-attn \
       --optimizer adam \
       "

INITIALIZATION_ARGS=" \
       --seed 42 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr 0.00002 \
       --lr-decay-style constant \
       --lr-warmup-fraction 0.0 \
       --min-lr 0.000001 \
       "

CHECKPOINTING_ARGS=" \
       --save-interval 1000 \
       --load ${LOAD_PATH} \
       --no-load-optim \
       --no-load-rng \
       --no-save-optim \
       --no-save-rng \
       "
# --pretrained-checkpoint /storage/huangyufei/pretrained_models/llama_megatron/megatron_ckpt_${MODEL_SIZE}B_${TP}_${PP}_${DP}

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --attention-softmax-in-fp32 \
       --no-query-key-layer-scaling \
       --attention-softmax-in-fp32 \
       "

DATA_ARGS=" \
       --train-data ${DATA_PATH} \
       --valid-data ${VALID_DATA_PATH} \
       --seq-length 2048 \
       --seq-length-train 1024 \
       --pad-to-max-length \
       --num-workers 2 \
       --tokenizer-type MixedTokenizer \
       --tokenizer-model ${STORAGE_PATH}/pretrained_models/llama_chinese/tokenizer_llama_en.model \
       --tokenizer-file ${STORAGE_PATH}/pretrained_models/llama_chinese/tokenizer_llama_zh.json \
       "

VALIDATION_ARGS=" \
       --eval-interval 10 \
       --eval-iters 10 \
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
       ${DATA_ARGS} \
       ${VALIDATION_ARGS} \
       "
# export CUDA_VISIBLE_DEVICES=1,2,3,5
$CMD 2>&1 | tee ${LOG_PATH}