import json

belle_500k_path = "/storage/lyy/codefile/instruct/instruct_dep_format.jsonl"
output_path = "/storage/huangyufei/Megatron-LM/datas/belle_497k_zh.json"

if __name__ == "__main__":
    belle_data = []
    with open(belle_500k_path) as f:
        for linenum, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            line_json = json.loads(line)
            belle_data.append(line_json)
    print(f"load {len(belle_data)} data from {belle_500k_path}")
    new_belle_data = []
    for d in belle_data:
        new_d = {}
        if 'instruction' in d:
            new_d['instruction'] = d['instruction']
            new_d['input'] = d['input']
        else:
            new_d['instruction'] = d['input']
            new_d['input'] = ""
        if 'target' in d:
            new_d['output'] = d['target']
        else:
            new_d['output'] = d['output']
        if new_d['output'] != "":
            new_belle_data.append(new_d)
    print(f"Dump {len(new_belle_data)} data to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(new_belle_data, f, indent=4, ensure_ascii=False)
        
    
    