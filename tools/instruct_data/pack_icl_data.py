import json
import sys
sys.path.append("/mnt/user/huangyufei/Megatron-LM")
from megatron.tokenizer.tokenizer import _GPTSentencePieceTokenizer
from tqdm import tqdm

source_filepath = "/mnt/public/data/sft/processed/data_0629/zh/processed/result_2_train.jsonl"
output_filepath = "/mnt/user/huangyufei/Megatron-LM/datas/DLM-2-SFT/zh/coigv2_10w_train.json"
tokenizer = _GPTSentencePieceTokenizer("/mnt/user/huangyufei/Megatron-LM/datas/chinese_llama.model")

pack_format = {
    "before_user": "### 用户(User):\n",
    "before_assistant": "### 助手(Assistant):\n"
}
max_length = 4080

if __name__ == "__main__":
    datas = []
    with open(source_filepath) as f:
        for liennum, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            line_json = json.loads(line)
            if line_json['index'] != 999999:
                datas.append(line_json)
            else:
                break
    print("datas len: ", len(datas))
    index = 0
    new_datas = []
    data_number = []
    pbar = tqdm(total=len(datas), desc='Packing data')
    exceed_number = 0
    while index < len(datas):
        current_length = 0
        current_number = 0
        new_data = {'instruction': '', 'output': ''}
        while index < len(datas):
            data = datas[index]
            input_str = data['instruction']
            output_str = data['output']
            total_str = pack_format['before_user'] + input_str + '\n' + pack_format["before_assistant"] + output_str + '\n'
            tokenized_str = tokenizer.tokenize(total_str, bos=False, eos=False)
            length = len(tokenized_str)
            current_length += length
            if current_length > max_length and new_data['instruction'] != '':
                break
            # add to new_data
            if new_data['instruction'] == '':
                new_data['instruction'] = input_str
            else:
                new_data['output'] += ("</s>" + input_str + "</s>")
            if current_length > max_length:
                exceed_number += 1
            new_data['output'] += output_str
            current_number += 1
            index += 1
            pbar.update(1)
        new_datas.append(new_data)
        data_number.append(current_number)
    print(f"pack {len(datas)} to {len(new_datas)}, average shot: {sum(data_number) / len(data_number)}, exceed number: {exceed_number}")
    with open(output_filepath, 'w') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)
        
        