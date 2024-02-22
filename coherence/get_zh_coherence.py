import os
import sys
import json

import torch
import numpy as np

import sentencepiece as spm

from datetime import datetime

from transformers import AutoModel, AutoModelForCausalLM

sys.path.append('/cpfs01/shared/public/lvkai/workspace/collie/')

from collie import CollieConfig, setup_distribution, env

data_root = "/cpfs01/shared/public/wanjuan_filter_32k/"
meta_root = "/cpfs01/shared/public/wanjuan_filter_32k-coherence/"

domain_list = ['ChinaNews-cn', 'Law-cn', 'Patent-cn', 'TextBook-cn', 'Wiki-cn', 'WebText-cn', ]

os.makedirs(data_root, exist_ok=True)
for domain in domain_list:
    os.makedirs(meta_root + domain, exist_ok=True)

path_dict = {domain: sorted(os.listdir(data_root + domain)) for domain in domain_list}
path_dict = {domain: [data_root + domain + '/' + file_name for file_name in path_dict[domain]] for domain in domain_list}

path_list = []
for domain in domain_list:
    for data_path in path_dict[domain]:
        path_list.append((domain, data_path))

model_path = '/cpfs01/user/liuxiaoran/llm/internlm2-boost-hf/7B/'

config = CollieConfig.from_pretrained(model_path, trust_remote_code=True)
config.tp_size = 1
config.pp_size = 1

setup_distribution(config=config)

model_config = config.model_config
model_config.__setattr__('_flash_attn_2_enabled', True)
try:
    model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config, torch_dtype=torch.float16, 
                                                 trust_remote_code=True).cuda()
except ValueError:
    model = AutoModel.from_pretrained(model_path, config=model_config, torch_dtype=torch.float16, 
                                      trust_remote_code=True).cuda()
model.eval()

enc_sp = spm.SentencePieceProcessor()
enc_sp.load(f'{model_path}tokenizer.model')

rank = env.rank
size = env.world_size

if env.rank == 0:
    print(f'rank = {rank}, size = {size}', flush=True)

path_list = path_list[rank::size]

max_context, med_context, min_context = 4096, 2048, 1024
batch_size, pad_token_id, vocab_size = 8, config.model_config.pad_token_id, config.model_config.vocab_size
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')

with torch.no_grad():

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
            input_ids_list = enc_sp.encode(data_dict['content'])

            len_token_list = len(input_ids_list)
            len_token_set = len(set(input_ids_list))

            comp_rate = len(input_ids_list) / len(data_dict['content'])
            dedu_rate = len(set(input_ids_list)) / len(input_ids_list)

            chunk_num = len(input_ids_list) // max_context 
            batch_num = int(np.ceil(chunk_num / batch_size))
            input_ids_list = torch.tensor(input_ids_list[: chunk_num * max_context])

            try:

                input_ids = input_ids_list.reshape((chunk_num, max_context)).to(dtype=torch.int64)
                attention_mask = (input_ids != pad_token_id)

                acc_wi, ppl_wi = [], []
                for i in range(batch_num):
                    logits = model(input_ids=input_ids[i*batch_size:(i+1)*batch_size, :].cuda(), 
                        attention_mask=attention_mask[i*batch_size:(i+1)*batch_size, :].cuda()).get('logits')[:, -min_context-1:-1]
                    labels = input_ids[i*batch_size:(i+1)*batch_size, -min_context:].cuda()
                    loss = loss_fn(logits.reshape(-1, vocab_size), labels.reshape(-1))
                    loss = torch.mean(loss.reshape((-1, min_context)).float(), dim=-1)
                    pred = torch.max(logits, dim=-1)[1]

                    acc_wi.append(torch.mean((pred == labels).float(), dim=-1))
                    ppl_wi.append(torch.exp(loss))

                acc_wi = torch.mean(torch.cat(acc_wi, dim=0)).item()
                ppl_wi = torch.mean(torch.cat(ppl_wi, dim=0)).item()

                input_ids = input_ids_list.reshape((chunk_num, max_context))[:,-med_context:].to(dtype=torch.int64)
                attention_mask = (input_ids != pad_token_id)

                acc_wo, ppl_wo = [], []
                for i in range(batch_num):
                    logits = model(input_ids=input_ids[i*batch_size:(i+1)*batch_size, :].cuda(), 
                        attention_mask=attention_mask[i*batch_size:(i+1)*batch_size, :].cuda()).get('logits')[:, -min_context-1:-1]
                    labels = input_ids[i*batch_size:(i+1)*batch_size, -min_context:].cuda()
                    loss = loss_fn(logits.reshape(-1, vocab_size), labels.reshape(-1))
                    loss = torch.mean(loss.reshape((-1, min_context)).float(), dim=-1)
                    pred = torch.max(logits, dim=-1)[1]

                    acc_wo.append(torch.mean((pred == labels).float(), dim=-1))
                    ppl_wo.append(torch.exp(loss))

                acc_wo = torch.mean(torch.cat(acc_wo, dim=0)).item()
                ppl_wo = torch.mean(torch.cat(ppl_wo, dim=0)).item()

                save_dct[str(data_dict['id'])] = {
                    'file_path': data_path, 'offset': offset, 
                    'comp_rate': comp_rate, 'dedu_rate': dedu_rate, 
                    'len_token_list': len_token_list, 'len_token_set': len_token_set, 
                    'acc_wi': float(acc_wi), 'acc_wo': float(acc_wo), 'acc_delta': float(acc_wi - acc_wo), 'acc_ratio': float((acc_wi - acc_wo) / (acc_wi + 1e-5)), 
                    'ppl_wi': float(ppl_wi), 'ppl_wo': float(ppl_wo), 'ppl_delta': float(ppl_wo - ppl_wi), 'acc_ratio': float((ppl_wo - ppl_wi) / (ppl_wo + 1e-5)), 
                    }
                
            except:
                print(f"[llm_score error] rank={env.rank}, (path_idx, offset)=({path_idx}, {offset})", flush=True)

                save_dct[str(data_dict['id'])] = {
                    'file_path': data_path, 'offset': offset, 
                    'comp_rate': comp_rate, 'dedu_rate': dedu_rate, 
                    'len_token_list': len_token_list, 'len_token_set': len_token_set, 
                    'acc_wi': -1, 'acc_wo': -1, 'acc_delta': -1, 'acc_ratio': -1, 
                    'ppl_wi': -1, 'ppl_wo': -1, 'ppl_delta': -1, 'acc_ratio': -1, 
                    }

            torch.cuda.empty_cache()
                        
            if rank == 0 and item_idx % 100 == 0:
                print('[{}]\t {}\t {}\t {} / {}'.format(datetime.now().time(), 
                    domain, item_idx, path_idx, len(path_list) ), flush=True)
                
        with open(meta_path, 'w+') as fp:
            json.dump(save_dct, fp, indent=4)

# # srun -p llm_o --quotatype=spot --ntasks-per-node=8 --ntasks=8 --cpus-per-task=1 python -u coherence/get_zh_coherence.py.py
