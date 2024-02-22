import os
import json

import numpy as np

from datetime import datetime

data_root = '/cpfs01/shared/public/slimpajama_filter_32k/'
meta_root = '/cpfs01/shared/public/slimpajama_filter_32k-complexity/'

domain_list = ['RedPajamaCommonCrawl', 'RedPajamaC4', 'RedPajamaArXiv', 'RedPajamaBook', 
               'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaWikipedia']

os.makedirs(data_root, exist_ok=True)
for domain in domain_list:
    os.makedirs(meta_root + domain, exist_ok=True)

path_dict = {domain: sorted(os.listdir(data_root + domain)) for domain in domain_list}
path_dict = {domain: [data_root + domain + '/' + file_name for file_name in path_dict[domain]] for domain in domain_list}

path_list = []
for domain in domain_list:
    for data_path in path_dict[domain]:
        path_list.append((domain, data_path))
        
# print(len(path_list))  # 224

rank = int(os.environ["SLURM_PROCID"])
size = int(os.environ["SLURM_NTASKS"])

path_list = path_list[rank::size]

for path_idx, (domain, data_path) in enumerate(path_list):
        
    save_dct, item_idx = {}, 0

    meta_path = data_path.replace(data_root, meta_root).replace('.jsonl', '.json')
    data_file = open(data_path, mode='r')  
    
    while True:
        
        try:
            offset = data_file.tell()
            data_line = data_file.readline()
            data_dict = json.loads(data_line)
        except:
            break
        
        item_idx += 1 
        content = data_dict['content'].lower()
        len_char, len_byte = len(content), len(content.encode('utf-8'))
        
        lines = [line for line in content.split('\n') if line.strip() != '']
        len_line = [len(line) for line in lines]
        mean, std, num = np.mean(len_line), np.std(len_line), len(len_line)
        upper, lower = max(len_line), min(len_line)
        median = np.median(len_line)
                
        enter_num = content.count('\n')
        comma_num = content.count(',')
        space_num = content.count(' ')
        bar_num = content.count('|')

        save_dct[str(data_dict['id'])] = {  # item_idx
            'file_path': data_path, 'offset': offset, 
            'len_char': len_char, 'num_line': num, 
            'len_line_mean': mean, 'len_line_std': std, 
            'len_line_med': median, 'len_line_max': upper, 'len_line_min': lower,
            'enter_num': enter_num, 'enter_rate': enter_num / len_char, 
            'comma_num': comma_num, 'comma_rate': comma_num / len_char, 
            'space_num': space_num, 'space_rate': space_num / len_char, 
            'bar_num': bar_num, 'bar_rate': bar_num / len_char, }    
                
        if rank == 0 and item_idx % 100 == 0:
            print('[{}]\t {}\t {}\t {} / {}'.format(datetime.now().time(), 
                domain, item_idx, path_idx, len(path_list) ), flush=True)
            
    with open(meta_path, 'w+') as fp:
        json.dump(save_dct, fp, indent=4)

# srun -p llm_t --ntasks-per-node=64 --ntasks=64 --cpus-per-task=1 python -u complexity/get_en_complexity.py
