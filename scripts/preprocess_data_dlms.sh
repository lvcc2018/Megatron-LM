python3 /mnt/user/lvchuancheng/DLM-4/Megatron-LM/tools/preprocess_data/preprocess_data_dlms.py \
--input ${1} \
--tokenizer-type GPTSentencePieceTokenizer \
--tokenizer-model /mnt/public/checkpoint/tokenzier/SentencePieceTokenizer/zh_en_code_65M_bf_bpe_4.model \
--output-prefix ${2} \
--workers 64 \
--log-interval 1000