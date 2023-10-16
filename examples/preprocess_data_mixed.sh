#!/usr/bin/env bash

set -ex

RUN_DIR=`pwd`

# The input file name
FILENAME=${1}

DATA_DIR="<Path to the directory of the original json data file>"
TOKENIZER_DIR="<Path to the directory of the tokenizer model/file>"
TOKENIZER_MODEL="<Name of the tokenizer model>"
TOKENIZER_FILE="<Name of the tokenizer file>"
OUTPUT_DIR="<Path to the output directory>"

python -u ${RUN_DIR}/tools/preprocess_data_mixed.py \
--input ${DATA_DIR}/${FILENAME}.jsonl \
--json-keys text \
--tokenizer-type MixedTokenizer \
--tokenizer-model ${TOKENIZER_DIR}/${TOKENIZER_MODEL} \
--tokenizer-file ${TOKENIZER_DIR}/${TOKENIZER_FILE} \
--output-prefix ${OUTPUT_DIR}/${FILENAME} \
--workers 16 \
--chunk-size 5000 \
--log-interval 1000
