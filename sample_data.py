import json
import random
import math

with open('LLaMA-Factory/data/wiki_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total_len = len(data)

ratios = [0.01,0.03,0.05]
output_files = ['wiki_sampled_1pct.json','wiki_sampled_3pct.json','wiki_sampled_5pct.json']

for ratio, output_file in zip(ratios, output_files):
    sample_size = math.ceil(total_len * ratio)
    sampled_data = random.sample(data, sample_size)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)