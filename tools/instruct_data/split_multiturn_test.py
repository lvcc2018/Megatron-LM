import random
import argparse
import json
import os


before_user = "### 用户(User):\n"
before_assistant = "### 助手(Assistant):\n"

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None, help="path for data to spilt")
    parser.add_argument("--sample_number", type=int, default=1000, help="number of instances to validation")
    parser.add_argument("--seed", type=int, default=42)
    return parser

def filter_by_turn(datas, min_turn_number, max_turn_number):
    turn_ge_4_number = 0
    turn_ge_4_datas = []
    for d in datas:
        turn_number = (len(d['output']) + 1) // 2
        if turn_number >= min_turn_number and turn_number <= max_turn_number:
            turn_ge_4_number += 1
            turn_ge_4_datas.append(d)
    print(f"Dialog Turns >= {min_turn_number} && <= {max_turn_number}: {turn_ge_4_number}")
    return turn_ge_4_datas

def filter_by_content(datas):
    new_datas = []
    max_bye_number = 2
    for d in datas:
        current_bye_number = 0
        for o in d['output']:
            if '再见' in o:
                current_bye_number += 1
            elif 'bye' in o.lower():
                current_bye_number += 1
        if current_bye_number > max_bye_number:
            continue
        new_datas.append(d)
    print(f"Bye number <= {max_bye_number}: {len(new_datas)}")
    return new_datas


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    if args.filepath.endswith('.json'):
        with open(args.filepath) as f:
            datas = json.load(f)
    elif args.filepath.endswith('.jsonl'):
        datas = []
        with open(args.filepath) as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                datas.append(json.loads(line))
    for d in datas:
        d['output'] = d['output'].split('</s>')
    print(f"load {len(datas)} instances from {args.filepath}")
    assert args.sample_number < len(datas)
    datas = filter_by_turn(datas, 3, 5)
    datas = filter_by_content(datas)
    idxs = list(range(len(datas)))
    random.seed(args.seed)
    random.shuffle(idxs)
    sample_idxs = idxs[:args.sample_number]
    sample_idxs.sort()
    dev_sample_datas = []
    for index in sample_idxs:
        dev_sample_datas.append(datas[index])
    print(f"Sample {len(dev_sample_datas)} for TEST.")
    test_sample_datas = []
    test_turn_number = []
    for index, data in enumerate(dev_sample_datas):
        instruction = data['instruction']
        outputs = data['output']
        turn_number = (len(outputs) + 1) // 2
        assert turn_number > 1
        target_turn_index = random.randint(1, turn_number - 1)
        inputs = [before_user + instruction + '\n']
        for turn_index in range(0, target_turn_index):
            inputs.append(before_assistant + outputs[turn_index * 2] + '\n')
            inputs.append(before_user + outputs[turn_index * 2 + 1] + '\n')
        inputs.append(before_assistant)
        output = outputs[target_turn_index * 2]
        test_sample_datas.append({
            'inputs': inputs,
            'output': output,
            'source': data['source'],
        })
        turn_number = target_turn_index + 1
        test_turn_number.append(turn_number)
        # print(json.dumps(data, indent=4, ensure_ascii=False))
        # print(json.dumps(test_sample_datas[-1], indent=4, ensure_ascii=False))
        # if index == 3:
        #     quit()
    
    basename = '.'.join(os.path.basename(args.filepath).split('.')[:-1])
    if basename.endswith('_dev'):
        basename = basename[:-4]
    dirpath = os.path.dirname(args.filepath)
    testpath = os.path.join(dirpath, basename + '_test.json')
    with open(testpath, 'w') as f:
        json.dump(test_sample_datas, f, indent=4, ensure_ascii=False)
    print(f"Save {len(dev_sample_datas)} dev data to {testpath}")
