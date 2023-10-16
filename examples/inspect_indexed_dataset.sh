
DATA_PATH="/mnt/user/gaocheng/task_1_1/Megatron-LM/Megatron-LM/data/cc_corpus_for_llama_text_document"
# DATA_PATH="0.1667-0.4444-0.32 zh /mnt/data/dlm_data/zh/zh_text_document" # zh: 380,098,113,596
# DATA_PATH+=" 0.5000-0.2222-0.32 en /mnt/data/dlm_data/en/en_text_document" # en: 241,235,656,716
# DATA_PATH+=" 0.3333 code /mnt/data/dlm_data/code/code_text_document" # code: 350,855,343,869
TOKENIZER_PATH=/mnt/public/landing/task1.2/tokenizer.model

CMD="python tools/inspect_indexed_dataset.py "
CMD+="--data-path ${DATA_PATH} "
CMD+="--data-cache-path data/ "
CMD+="--split 1000,0,0 "
CMD+="--seq-length 8192 "
CMD+="--global-batch-size 768 "
CMD+="--train-iters 151000 "
CMD+="--tokenizer-type GPTSentencePieceTokenizer "
CMD+="--tokenizer-model ${TOKENIZER_PATH} "
CMD+="--data-impl mmap "
CMD+="--use-dataset-manager "
CMD+="--seed 1403 "
# CMD+="--iteration-start 206 "
# CMD+="--iteration-end 206 "
# CMD+="--index-start 768 "
# CMD+="--index-end 769 "
CMD+="--doc-ids 1,2-4 "
# CMD+="--show-iters "
CMD+="--output-dir log/inspect_data "

echo ${CMD}
${CMD}
