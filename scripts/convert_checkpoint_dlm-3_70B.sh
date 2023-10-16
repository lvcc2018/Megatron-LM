#! /bin/bash

python3 tools/convert_checkpoint/megatron_llama2.py \
--megatron-path ${1} \
--load-path ${2} \
--save-path ${3} \
--addition-vocab-size 0 \
--print-checkpoint-structure \
--target_tensor_model_parallel_size 4 \
--target_pipeline_model_parallel_size 4 \
--target_params_dtype bf16 \
--make_vocab_size_divisible_by 1