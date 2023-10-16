import json
from tqdm import tqdm

filepath = "/mnt/data01/huangyufei/Megatron-LM/datas/sixth_preprocess/zh/zh_multi_turn.jsonl"
outputpath = "/mnt/data01/huangyufei/Megatron-LM/datas/sixth_preprocess/zh/zh_multi_turn_2.jsonl"

if __name__ == "__main__":
    datas = []
    with open(filepath) as f:
        for linenum, line in enumerate(tqdm(f)):
            line = line.strip()
            if line == '':
                continue
            line_json = json.loads(line)
            output = line_json['output'].split('\s')
            output = '</s>'.join(output)
            line_json['output'] = output
            datas.append(line_json)
    with open(outputpath, 'w') as f:
        for d in tqdm(datas, desc="writing"):
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    