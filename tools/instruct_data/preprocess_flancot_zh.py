import json


filepath = "/mnt/data01/huangyufei/Megatron-LM/datas/rawdata/CoT_data.json"
outputpath = "/mnt/data01/huangyufei/Megatron-LM/datas/second_preprocess/flancot_zs_79k_en.json"

if __name__ == "__main__":
    with open(filepath) as f:
        datas = json.load(f)
    print(f"load {len(datas)} data")
    for d in datas:
        d['instruction'] = d['instruction'].replace("\\n", "\n")
        d['output'] = d['output'].replace("\\n", "\n")
    with open(outputpath, 'w') as f:
        json.dump(datas, f, indent=4, ensure_ascii=False)
    print(f"save to {outputpath}")
