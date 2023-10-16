import json

filepath = "/mnt/data01/huangyufei/Megatron-LM/datas/fifth_preprocess/merged_ZBenchdata_sheet1_zh.jsonl"
outputpath = "/mnt/data01/huangyufei/Megatron-LM/datas/fifth_preprocess/merged_ZBenchdata_sheet1_zh.json"

if __name__ == "__main__":
    datas = []
    with open(filepath, 'r') as f:
        for linenum, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            line_json = json.loads(line)
            datas.append(line_json)
    print(f"load {len(datas)} data from {filepath}")
    for d in datas:
        instruction = d.pop('instruct')
        d['instruction'] = instruction
    with open(outputpath, 'w') as f:
        json.dump(datas, f, indent=4, ensure_ascii=False)
    print(f"save to {outputpath}")
    
            