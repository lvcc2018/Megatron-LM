# !/bin/bash
# created by Yufei Huang

BASE_PATH="/mnt/data/huangyufei/Megatron-LM"

python ${BASE_PATH}/tools/process_chunk.py \
    --data_path /mnt/data/data/stage_0/code/code_1800G_text_document \
    --data_name code \
    --split_name train \
    --splits_string 1000,0,0 \
    --data_impl infer \
    --max_seq_length 4096 \
    --batch_size 1536 \
    --num_samples 81918672 \
    --full_doc_ratio 0.5 \
    --buffer_size 1000 \
    --tokenizer_model /mnt/data/lvchuancheng/checkpoint/tokenizer/chinese_llama.model \
    --seed 1403 \
    --multiprocess_chunksize 20000 \
    --multiprocess_num 20 \
    --data_cache_dir ${BASE_PATH}/datas
    
    