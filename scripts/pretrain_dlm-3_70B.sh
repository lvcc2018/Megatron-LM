#! /bin/bash
# Pretraining DLM-3 70B model, Good Luck!

# Setting the environment variables
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO

# Distributed training variables
GPUS_PER_NODE=8

# Parallelism variables
TP=4
PP=4

# Network size variables
MODEL_SIZE=70

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DROP_OUT=0.1
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN=4096

NAME=pretrain

DATA_PATH=<PATH_TO_DATA>

LOG_NAME=DLM_${MODEL_SIZE}b_WS${WORLD_SIZE}_TP${TP}_PP${PP}_MAXLR${MAX_LR}_DROP${DROP_OUT}_${NAME}

SRC_PATH=<PATH_TO_SRC>/Megatron-LM/pretrain_gpt.py
SAVE_PATH=<PATH_TO_SAVE>/checkpoint/${LOG_NAME}
LOAD_PATH=<PATH_TO_LOAD>
LOG_PATH=<PATH_TO_LOG>/log/${LOG_NAME}_${RANK}.log
TOKENIZER_PATH=<PATH_TO_TOKENIZER>

# wandb environment
export WANDB_API_KEY="e0b30216258c751235154d145c5deab25d92f7b3"
export WANDB_PROJECT="Pretrain"
export WANDB_ENTITY="deeplang-ai"
export WANDB_NAME="${LOG_NAME}"
export WANDB_NOTES="DLM-3 70B Pretraining"

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${WORLD_SIZE} \
       --node_rank ${RANK} \
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
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --position-embedding-type rope \
       --max-position-embeddings 4096 \
       --make-vocab-size-divisible-by 1 \
       --layernorm-epsilon ${NORM_EPS} \
       --normalization rmsnorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
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
       --global-batch-size 1024 \
       --train-iters 10000 \
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
       --lr-warmup-iters 580 \
       --min-lr ${MIN_LR} \
       "

CHECKPOINTING_ARGS=" \
       --load ${LOAD_PATH} \
       --save ${SAVE_PATH} \
       --save-interval 100 \
       "

MIXED_PRECISION_ARGS=" \
       --bf16 \
       --no-query-key-layer-scaling \
       "

# VALIDATION_ARGS=" \
#        --eval-interval 100 \
#        --eval-iters 10 \
#        "

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