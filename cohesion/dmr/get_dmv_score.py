import os
import json
import sys
import torch
import numpy as np

from datetime import datetime

from dmv_inference import DMVModel, AutoTokenizer

from collie import CollieConfig, setup_distribution, env
from ltp import StnSplit

ltp_sp = StnSplit()


def sentence_tokenize(doc: str):
    sentences = ltp_sp.split(doc)
    sentences = [i for i in sentences if len(i) >= 5]
    return sentences

data_root = '/cpfs01/shared/public/wanjuan_filter_32k/'
meta_root = '/cpfs01/shared/public/wanjuan_filter_32k-dmv_score_prob/'

domain_list = ['ChinaNews-cn', 'Law-cn', 'Patent-cn', 'TextBook-cn', 'WebText-cn', 'Wiki-cn']

os.makedirs(data_root, exist_ok=True)
for domain in domain_list:
    os.makedirs(meta_root + domain, exist_ok=True)

path_dict = {domain: sorted(os.listdir(data_root + domain)) for domain in domain_list}
path_dict = {domain: [data_root + domain + '/' + file_name for file_name in path_dict[domain]] for domain in
             domain_list}

path_list = []
for domain in domain_list:
    for data_path in path_dict[domain]:
        path_list.append((domain, data_path))

ckpt_path = '/cpfs01/shared/public/dmv/save_zh/K_141_EM_500_bsz_64_psi_3e-05_phi_0.01/em-conn.pt'
plm_name_or_path = '/cpfs01/shared/public/dmv/save_zh/K_141_EM_500_bsz_64_psi_3e-05_phi_0.01/em-roberta'

config = CollieConfig.from_pretrained(plm_name_or_path)

setup_distribution(config=config)

rank = env.rank
size = env.world_size

path_list = path_list[rank::size]

model = DMVModel(plm_name_or_path, C=141, K=141).cuda()
ckpt = torch.load(ckpt_path)
if rank == 0:
    print(f'loading model with test-acc {ckpt["test-acc"]}')
    # loading model with test-acc 0.45517730712890625
model.load_state_dict(ckpt['weights'], strict=False)

tokenizer = AutoTokenizer.from_pretrained(plm_name_or_path)

max_length, batch_size = 128, 256

if env.rank == 0:
    print(f'rank = {rank}, size = {size}', flush=True)

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
            content = data_dict['content']

            raw_sents = sentence_tokenize(content)
            num_raw_sent = len(raw_sents)

            args, dmv_score, tmp_sent = [], [], ''

            for sent in raw_sents:
                tmp_sent = sent if tmp_sent == '' else tmp_sent + ' ' + sent
                if len(tmp_sent) > max_length * 2:
                    args.append(tmp_sent)
                    tmp_sent = ''
            if tmp_sent != '':
                args.append(tmp_sent)

            if len(args) > 1:
                arg1s, arg2s = args[:-1], args[1:]
            else:
                arg1s, arg2s = args, ['']

            batch_num, num_cat_sent = int(np.ceil(len(arg1s) / batch_size)), len(args)

            for i in range(batch_num):
                u_dict = tokenizer(arg1s[i * batch_size:(i + 1) * batch_size], return_tensors='pt',
                                   padding=True, truncation=True, max_length=max_length).to(torch.device("cuda"))
                v_dict = tokenizer(arg2s[i * batch_size:(i + 1) * batch_size], return_tensors='pt',
                                   padding=True, truncation=True, max_length=max_length).to(torch.device("cuda"))
                ret = model(u_dict, v_dict)
                logp_c = ret['logp_c']
                p_c = torch.exp(logp_c)
                dmv_score.append(p_c)
            dmv_score = torch.cat(dmv_score)

            save_dct[str(item_idx)] = {
                'file_path': data_path, 'offset': offset,
                'num_raw_sent': num_raw_sent, 'num_cat_sent': num_cat_sent,
                'dmv_max': torch.max(dmv_score, dim=0)[0].tolist(), 'dmv_std': torch.std(dmv_score, dim=0).tolist(),
                'dmv_mean': torch.mean(dmv_score, dim=0).tolist(),
            }

            if item_idx % 100 == 0:
                print('[{}]\t {}\t {}\t {} / {}'.format(datetime.now().time(),
                                                        domain, item_idx, path_idx, len(path_list)), flush=True)

        with open(meta_path, 'w+') as fp:
            json.dump(save_dct, fp, indent=4)
