import json
from tqdm import tqdm

filepath = "/nfs/huangyufei/Megatron-LM/datas/sixth_preprocess/zh_alpaca_gpt4_belle_manual_train.json"
filepath = "/mnt/data01/huangyufei/Megatron-LM/datas/sixth_preprocess/zh/zh_alpaca_gpt4_belle_manual_dev.json"
# filepath = "/nfs/huangyufei/Megatron-LM/datas/sixth_preprocess/zh_alpaca_gpt4_belle_manual.jsonl"
keyword = "鸡和兔子"

def load_data(path):
    print(f"loading from {filepath} ...")
    if path.endswith(".json"):
        with open(path) as f:
            datas = json.load(f)
    elif path.endswith(".jsonl"):
        datas = []
        with open(path) as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == "":
                    continue
                datas.append(json.loads(line))
    print(f"load {len(datas)} data from {filepath}")
    return datas

if __name__ == "__main__":
    datas = load_data(filepath)
    cnt = 0
    for d in tqdm(datas):
        if keyword in d['instruction'] or keyword in d['output']:
            print(f"Cnt: {cnt}")
            print(json.dumps(d, indent=4, ensure_ascii=False))
            cnt += 1
    