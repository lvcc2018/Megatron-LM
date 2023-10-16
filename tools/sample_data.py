import argparse
import random
import json
import os
from tqdm import tqdm

def list_files_in_directory(directory):
    file_names = []
    for entry in os.scandir(directory):
        if entry.is_file():
            file_names.append(entry.name)
        else:
            file_names.extend(list_files_in_directory(entry))
    return file_names

def convert_chunk_to_jsonl(file_name, save_name):
    data = json.load(open(file_name, 'r'))
    with open(save_name, 'w') as outfile:
        for line in data:
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write('\n')

def get_bypes_num(string):
    return len(string.encode())

def sample_data(path, target_gb, save_name):
    file_names = list_files_in_directory(path)
    target_size = target_gb * (10**9)
    all_size = 0
    current_size = 0
    all_data = {}
    res_data = {file_name:{'current_size':0,'all_size':0, 'res_lines':[]} for file_name in file_names}
    for file_name in tqdm(file_names):
        all_data[file_name] = open(os.path.join(path, file_name), 'r')
        bytes_num = os.path.getsize(os.path.join(path, file_name))
        res_data[file_name]['all_size'] = bytes_num
        all_size += bytes_num
    ratio_dic = {file_name:res_data[file_name]['all_size']/all_size for file_name in file_names}
    print(res_data)
    print(ratio_dic)
    print(file_names)
    print([ratio_dic[file_name] for file_name in file_names])
    if target_size > all_size:
        print('target_size is larger than all_size')
        return
    print("Trying to sample {} GB from {} GB".format(target_gb, all_size / (10**9)))
    while current_size < target_size:
        f_n = random.choices(file_names, weights=[ratio_dic[file_name] for file_name in file_names], k=1)[0]
        line = all_data[f_n].readline()
        if line != '':
            line = json.loads(line.strip())
            bytes_num = get_bypes_num(line['text'])
            res_data[f_n]['current_size'] += bytes_num
            res_data[f_n]['res_lines'].append(line)
            current_size += bytes_num
    final_data = []
    for file_name in file_names:
        final_data.extend(res_data[file_name]['res_lines'])
    with open(save_name, 'w') as outfile:
        for line in final_data:
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write('\n')
    print('Finally get {} GB data and saved.'.format(current_size / (10**9)))
    for file_name in file_names:
        print('{}: {} GB'.format(file_name, res_data[file_name]['current_size'] / (10**9)))

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])

def sample_data_by_line(path, target_gb, save_name):
    file_names = list_files_in_directory(path)
    target_size = target_gb * (10**9)
    all_size = 0
    for file_name in tqdm(file_names):
        bytes_num = os.path.getsize(os.path.join(path, file_name))
        all_size += bytes_num
    ratio = target_size / all_size
    print("Trying to sample {} GB from {} GB".format(target_gb, all_size / (10**9)))
    res_gb = {}
    outfile = open(save_name, 'w')
    for file_name in tqdm(file_names):
        line_num = wc_count(os.path.join(path, file_name))
        sample_num = int(line_num * ratio)
        res_gb[file_name] = 0
        f = open(os.path.join(path, file_name), 'r')
        for i in range(sample_num):
            line = f.readline()
            line = json.loads(line.strip())
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write('\n')
            res_gb[file_name] += get_bypes_num(line['text'])
    print('Finally get {} GB data and saved.'.format(target_size / (10**9)))
    for file_name in file_names:
        print('{}: {} GB ({} %)'.format(file_name, res_gb[file_name] / (10**9)), res_gb[file_name] / target_size * 100)

def merge_and_save_to_jsonl(path, save_name):
    file_names = list_files_in_directory(path)
    data = []
    for file_name in tqdm(file_names):
        if 'chunk' in file_name:
            data.extend(json.load(open(os.path.join(path, file_name), 'r')))
        else:
            data.extend([json.loads(line.strip()) for line in open(os.path.join(path, file_name), 'r').readlines()])
    with open(save_name, 'w') as outfile:
        for line in data:
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample a JSONL file")
    parser.add_argument("--path", help="Path to the input JSONL files")
    parser.add_argument("--target_gb", type=float, help="Target data size in GB")
    parser.add_argument("--save_name", help="Path to the output JSONL file")

    args = parser.parse_args()
    sample_data_by_line(args.path, args.target_gb, args.save_name)
    merge_and_save_to_jsonl(args.path, args.save_name)