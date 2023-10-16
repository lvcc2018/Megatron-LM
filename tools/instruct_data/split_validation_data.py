import random
import argparse
import json
import os


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None, help="path for data to spilt")
    parser.add_argument("--sample_number", type=int, default=1000, help="number of instances to validation")
    parser.add_argument("--seed", type=int, default=42)
    return parser


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
    print(f"load {len(datas)} instances from {args.filepath}")
    assert args.sample_number < len(datas)
    idxs = list(range(len(datas)))
    random.seed(args.seed)
    random.shuffle(idxs)
    sample_idxs = idxs[:args.sample_number]
    sample_idxs.sort()
    dev_sample_datas = []
    for index in sample_idxs:
        dev_sample_datas.append(datas[index])
    print(f"Sample {len(dev_sample_datas)} for dev.")
    sample_idxs = idxs[args.sample_number:]
    sample_idxs.sort()
    train_sample_datas = []
    for index in sample_idxs:
        train_sample_datas.append(datas[index])
    print(f"sample {len(train_sample_datas)} from train.")
    basename = '.'.join(os.path.basename(args.filepath).split('.')[:-1])
    dirpath = os.path.dirname(args.filepath)
    devpath = os.path.join(dirpath, basename + '_dev.json')
    trainpath = os.path.join(dirpath, basename + '_train.json')
    with open(devpath, 'w') as f:
        json.dump(dev_sample_datas, f, indent=4, ensure_ascii=False)
    print(f"Save {len(dev_sample_datas)} dev data to {devpath}")
    with open(trainpath, 'w') as f:
        json.dump(train_sample_datas, f, indent=4, ensure_ascii=False)
    print(f"Save {len(train_sample_datas)} train data to {trainpath}")
    
    
    