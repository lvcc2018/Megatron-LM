#! /bin/bash

python3 tools/preprocess_data/preprocess_data_llama2.py \
--input ${1} \
--tokenizer-type GPTSentencePieceTokenizer \
--tokenizer-model ${2} \
--output-prefix ${3} \
--workers 1 \
--partitions 1 \
--log-interval 1000