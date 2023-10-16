#! /bin/bash
killall python
# Continue training the LLaMA model
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

# Distributed training variables
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR="10.119.53.209"
MASTER_PORT=6002
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))

# Parallelism variables
TP=1
PP=1
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
DROP_OUT=0.1
MAX_LR=6e-5
MIN_LR=6e-6
NAME=zh45_en30_code6-0.2315_0.6945_it0.25_0.074-2

# Paths and names
# LOG_NAME=llama_${MODEL_SIZE}b_WS${WORLD_SIZE}_TP${TP}_PP${PP}_MAXLR${MAX_LR}_DROP${DROP_OUT}_${NAME}
LOG_NAME=debug_pretrain
DATA_PATH="0.2315-0.6174-0.25 zh /mnt/data01/shenyan/data/DLM-2-data/7B/zh45_en30_code6/zh45/zh45_text_document" # 0.556
DATA_PATH+=" 0.6945-0.3086-0.25 en /mnt/data01/shenyan/data/DLM-2-data/7B/zh45_en30_code6/en30/en30_text_document" # 0.370
DATA_PATH+=" 0.074 code /mnt/data01/shenyan/data/DLM-2-data/7B/zh45_en30_code6/code6/code6_text_document" # 0.074
# DATA_PATH=/mnt/data01/shenyan/data/DLM-2-data/7B/zh45_en30_code6/mixed/mixed_text_document
SRC_PATH=/mnt/data01/huangyufei/Megatron-LM/pretrain_gpt.py
SAVE_PATH=/mnt/data01/huangyufei/Megatron-LM/checkpoint/${LOG_NAME}
LOAD_PATH=/mnt/data01/shenyan/lvcc/DLM-2/Exp_7B/checkpoints/origin_TP1_PP1_DP64_warmup
LOG_PATH=/mnt/data01/huangyufei/Megatron-LM/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p /mnt/data01/huangyufei/Megatron-LM/log/${LOG_NAME}
TSBD_PATH=/mnt/data01/huangyufei/Megatron-LM/tensorboard/${LOG_NAME}

SAVE_STEPS=10000
EVAL_STEPS=10

# wandb environment
export WANDB_API_KEY="yourapikey"
export WANDB_PROJECT="Pretrain"  # project name
export WANDB_ENTITY="deeplang"  # orgnization name
export WANDB_NAME="${LOG_NAME}" # this run name
export WANDB_NOTES="debug whether wandb can work" # Longer notes about your run. Markdown is allowed and you can edit this later in the UI.
# more environment variables: https://docs.wandb.ai/guides/track/environment-variables
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
       --finetune \
       --micro-batch-size 1 \
       --global-batch-size 128 \
       --train-iters 20000 \
       --log-interval 1 \
       --tensorboard-dir ${TSBD_PATH}
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --use-flash-attn \
       --optimizer adam \
       --exit-interval 100000 \
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
       "

CHECKPOINTING_ARGS=" \
       --save ${SAVE_PATH} \
       --save-interval ${SAVE_STEPS} \
       --load ${LOAD_PATH} \
       --no-load-optim \
       --no-load-rng \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --no-query-key-layer-scaling \
       "

VALIDATION_ARGS=" \
       --eval-interval ${EVAL_STEPS} \
       --eval-iters 10 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 998,1,1 \
       --seq-length 2048 \
       --num-workers 0 \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model /mnt/data01/shenyan/lvcc/DLM-2/Exp_7B/tokenizer/SentencePieceTokenizer/chinese_llama.model \
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
echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}