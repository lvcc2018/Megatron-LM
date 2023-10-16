from nltk.translate.bleu_score import sentence_bleu
import json
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from collections import Counter

gpt_turbo_data_path = ["/storage/huangyufei/Megatron-LM/datas/rawdata/instruct_alpaca_3_5_deduped.jsonl"]
gpt_4_data_path = ["/storage/huangyufei/Megatron-LM/datas/rawdata/instruct_gpt4_deduped.jsonl"]
outputfile = "/storage/huangyufei/Megatron-LM/datas/rawdata/instruct_gpt4_deduped_20k.jsonl"

def load_jsonl(path):
    datas = []
    with open(path) as f:
        for linenum, line in enumerate(f):
            line = line.strip()
            if line == "":
                continue
            line_json = json.loads(line)
            datas.append(line_json)
    print(f"load {len(datas)} data from {path}")
    return datas

def f1_score(prediction, ground_truth):
    prediction_tokens = list(prediction)
    ground_truth_tokens = list(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def find_similar_prompt(d, gpt4datas):
    prompt = d['input']
    output = d['output']
    # prompt_tokens = list(prompt + '\n' + output)
    prompt_tokens = list(prompt)
    bleu_scores = []
    for d4 in gpt4datas:
        prompt_4 = d4['input']
        output_4 = d4['output']
        # prompt_4_tokens = list(prompt_4 + '\n' + output_4)
        prompt_4_tokens = list(prompt_4)
        # bleu = sentence_bleu(prompt_4_tokens, prompt_tokens, weights=(1, 0, 0, 0))
        bleu = f1_score(prompt_4, prompt)
        bleu_scores.append(bleu)
    bleu_scores_indexes = list(enumerate(bleu_scores))
    bleu_scores_indexes.sort(key=lambda p: p[1], reverse=True)
    index = bleu_scores_indexes[0][0]
    score = bleu_scores_indexes[0][1]
    co_d4 = gpt4datas[index]
    return co_d4, score

if __name__ == "__main__":
    gpt_turbo_datas = []
    for path in gpt_turbo_data_path:
        gpt_turbo_datas.extend(load_jsonl(path))
    print(f"load {len(gpt_turbo_datas)} data for gpt-turbo")
    gpt_4_datas = []
    for path in gpt_4_data_path:
        gpt_4_datas.extend(load_jsonl(path))
    print(f"load {len(gpt_4_datas)} data for gpt4")
    select_gpt4_data = []
    pools = Pool(processes = 10)
    index = 0
    for d, score in tqdm(pools.imap(partial(find_similar_prompt, gpt4datas=gpt_4_datas), gpt_turbo_datas, chunksize=10), total=len(gpt_turbo_datas)):
        # print(f"index: {index}")
        # print("Gpt turbo:", gpt_turbo_datas[index])
        # print("GPT 4    :", d)
        # print("Score    :", score)
        # index += 1
        # if index == 15:
        #     quit()
        index += 1
        select_gpt4_data.append(d)
    print(f"select {len(select_gpt4_data)} datas, save to {outputfile}")
    with open(outputfile, 'w') as f:
        for d in select_gpt4_data:
            f.write(json.dumps(d) + '\n')
    print("done :)")
            
    