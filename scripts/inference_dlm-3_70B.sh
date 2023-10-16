#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

# This example will start serving the 345M model that is partitioned 8 way tensor parallel
DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=<PATH_TO_CHECKPOINT>
TOKENIZER_PATH=<PATH_TO_TOKENIZER>

# Network size variables
MODEL_SIZE=70

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

torchrun $DISTRIBUTED_ARGS tools/run_text_generation.py   \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers $NUM_LAYERS  \
       --hidden-size $HIDDEN_SIZE  \
       --ffn-hidden-size $FFN_HIDDEN_SIZE  \
       --num-attention-heads $NUM_HEAD \
       --group-query-attention \
       --num-query-groups ${NUM_KV_HEAD} \
       --position-embedding-type rope \
       --max-position-embeddings 4096 \
       --rotary-percent 1.0 \
       --make-vocab-size-divisible-by 1 \
       --layernorm-epsilon ${NORM_EPS} \
       --normalization rmsnorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --load ${CHECKPOINT}  \
       --tokenizer-type GPTSentencePieceTokenizer  \
       --tokenizer-model ${TOKENIZER_PATH} \
       --bf16  \
       --seq-length 4096  \
       --out-seq-length 1024  \
       --temperature 1.0  \
       --top_p 0.9  \
       --seed 42 \
       --micro-batch-size 1
