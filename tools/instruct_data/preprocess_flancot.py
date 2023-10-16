import json
from tqdm import tqdm


zero_shot_opt = "/storage/huangyufei/Megatron-LM/datas/rawdata/flan/cot_zs_noopt_train.jsonl"
zero_shot_noopt = "/storage/huangyufei/Megatron-LM/datas/rawdata/flan/cot_zs_opt_train.jsonl"
few_shot_opt = "/storage/huangyufei/Megatron-LM/datas/rawdata/flan/cot_fs_noopt_train.jsonl"
few_shot_noopt = "/storage/huangyufei/Megatron-LM/datas/rawdata/flan/cot_fs_opt_train.jsonl"
zero_output_path = "/storage/huangyufei/Megatron-LM/datas/flancot_zs_140k_en.json"
few_output_path = "/storage/huangyufei/Megatron-LM/datas/flancot_fs_140k_en.json"

def handle_one_file(filepath):
    datas = []
    with open(filepath) as f:
        for linenum, line in enumerate(f):
            line = line.strip()
            if line == "":
                continue
            line_json = json.loads(line)
            datas.append(line_json)
    print(f"load {len(datas)} data from {filepath}")
    return datas

def preprocess_datas(input_paths, output_path):
    datas = []
    for path in input_paths:
        datas.extend(handle_one_file(path))
    new_datas = []
    for d_index, d in enumerate(tqdm(datas, desc='Converting Flan CoT')):
        new_d = {}
        try:
            new_d['instruction'] = d['inputs']
        except TypeError:
            print(d_index, d)
            quit()
        new_d['input'] = ""
        output = d["targets"]
        if output == "":
            continue
        new_d['output'] = [d["targets"]]
        new_datas.append(new_d)
    print(f"Dump {len(new_datas)} data to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(new_datas, f, indent=4, ensure_ascii=False)

def inspect_datas():
    datas = handle_one_file(zero_shot_opt)
    example = 'Are more people today related to Genghis Khan than Julius Caesar?'
    print("Zero Shot Option:")
    for index, data in enumerate(datas):
        if example in data['inputs']:
            print(f"index :{index}")
            print(json.dumps(data, indent=4, ensure_ascii=False))
    datas = handle_one_file(zero_shot_noopt)
    print("Zero Shot No Option:")
    for index, data in enumerate(datas):
        if example in data['inputs']:
            print(f"index :{index}")
            print(json.dumps(data, indent=4, ensure_ascii=False))
    datas = handle_one_file(few_shot_noopt)
    print("Few Shot No Option:")
    for index, data in enumerate(datas):
        if example in data['inputs']:
            print(f"index :{index}")
            print(json.dumps(data, indent=4, ensure_ascii=False))
    datas = handle_one_file(few_shot_opt)
    print("Few Shot No Option:")
    for index, data in enumerate(datas):
        if example in data['inputs']:
            print(f"index :{index}")
            print(json.dumps(data, indent=4, ensure_ascii=False))
        

if __name__ == "__main__":
    inspect_datas()
    # preprocess_datas([zero_shot_opt, zero_shot_noopt], zero_output_path)
    # preprocess_datas([few_shot_opt, few_shot_noopt], few_output_path)