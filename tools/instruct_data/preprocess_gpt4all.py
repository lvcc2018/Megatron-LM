import json
from tqdm import tqdm

filepath = "/mnt/data01/huangyufei/Megatron-LM/datas/rawdata/gpt4all/raw_data_sanity_cleaned_without_p3/data.jsonl"
outputpath = "/mnt/data01/huangyufei/Megatron-LM/datas/gpt4all_en_without_p3_dialogue.json"

if __name__ == "__main__":
    datas = []
    with open(filepath) as f:
        for linenum, line in enumerate(f):
            line = line.strip()
            if line == "":
                continue
            line_json = json.loads(line)
            datas.append(line_json)
    print(f"load {len(datas)} data from {filepath}")
    # while True:
    #     index = input("Input Index:")
    #     index = int(index)
    #     print(json.dumps(datas[index], indent=4, ensure_ascii=False))
    # source_dict = {}
    # for d in datas:
    #     source = d['source']
    #     if source not in source_dict:
    #         source_dict[source] = 0
    #     source_dict[source] += 1
        
    # print(source_dict)
    # for d in datas:
    #     source = d['source']
    #     if source == "pacovaldez/stackoverflow-questions":
    #         print(json.dumps(d, indent=4))
    #         break
    
    # cnt = 0
    # for d in datas:
    #     source = d['source']
    #     if source == "unified_chip2":
    #         print(json.dumps(d, indent=4))
    #         cnt += 1
    #         if cnt == 3:
    #             break
    dialogue_datas = []
    for d in datas:
        source = d['source']
        if source == "laion/unified_chip2":
            dialogue_datas.append(d)
    datas = dialogue_datas
    
    # for d in datas:
    #     source = d['source']
    #     if source == "":
    #         print(json.dumps(d, indent=4))
    #         break
    
    new_datas = []
    for d in tqdm(datas, desc='Converting gpt4all'):
        source = d['source']
        instruction = d['prompt']
        # if source == "pacovaldez/stackoverflow-questions":
        #     instruction = instruction.replace("<p>", "")
        #     instruction = instruction.replace("</p>", "")
        #     instruction = instruction.replace("<pre>", "")
        #     instruction = instruction.replace("</pre>", "")
        #     instruction = instruction.replace("<code>", "")
        #     instruction = instruction.replace("</code>", "")
        #     instruction = instruction.replace("<blockquote>", "")
        #     instruction = instruction.replace("</blockquote>", "")
        #     instruction = instruction.replace("<h1>", "")
        #     instruction = instruction.replace("</h1>", "")
        # if source == "laion/unified_chip2":
        #     output = d['response'].split('</s>')
        # else:
        #     output = [d['response']]
        # output = [o.strip() for o in output]
        # output = [o for o in output if o != ""]
        # if len(output) == 0:
        #     continue
        output = d['response']
        new_d = {}
        new_d['instruction'] = instruction
        new_d['input'] = ""
        new_d['output'] = output
        new_datas.append(new_d)
        # if len(new_datas) == 3:
        #     quit()
    print(f"Dump {len(new_datas)} data to {outputpath}")
    with open(outputpath, 'w') as f:
        json.dump(new_datas, f, indent=4, ensure_ascii=False)
        
    