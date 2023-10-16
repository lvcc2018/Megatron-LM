#!/bin/bash
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

# Network size variables
MODEL_SIZE=70

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

CHECKPOINT=<PATH_TO_CHECKPOINT>
TOKENIZER_PATH=<PATH_TO_TOKENIZER>

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --load ${CHECKPOINT}  \
       --num-attention-heads ${NUM_HEAD} \
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --position-embedding-type rope \
       --max-position-embeddings 4096 \
       --layernorm-epsilon ${NORM_EPS} \
       --normalization rmsnorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --no-query-key-layer-scaling \
       --tokenizer-type GPTSentencePieceTokenizer \
       --seq-length 4096 \
       --micro-batch-size 1  \
       --out-seq-length 4096  \
       --temperature 1.0  \
       --tokenizer-model ${TOKENIZER_PATH} \
       --top_p 0.9  \
       --seed 42 \
       --make-vocab-size-divisible-by 1 \
       --attention-dropout 0 \
       --hidden-dropout 0