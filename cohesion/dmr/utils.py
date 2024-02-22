import torch
import json
from torch.utils.data import Dataset
from collections import Counter


class ConnData(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    @classmethod
    def from_datasets(cls, datum):
        dataset = []
        for data in datum:
            dataset.extend(data)

        return cls(dataset)


def get_joint_collate_fn(tokenizer, device=0):
    def collate_fn(examples):
        arg1s, arg2s, labels, rels = [], [], [], []
        for example in examples:
            arg1s.append(example['arg1'])
            arg2s.append(example['arg2'])
            labels.append(example['conn'])
            rels.append(example['rel'])
            
        u_dict = tokenizer(arg1s, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        v_dict = tokenizer(arg2s, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        labels = torch.LongTensor(labels).to(device)
        rels = torch.LongTensor(rels).to(device)
        
        input_dict = {
            'u_dict': u_dict,
            'v_dict': v_dict,
            'labels': labels,
            'rels': rels
        }
        
        return input_dict
    return collate_fn


def read_discovery_connectives(fpath):
    with open(fpath) as fp:
        data = json.load(fp)

    conn_counter = Counter()
    for split in ['train', 'dev', 'test']:
        for example in data[split]:
            conn_counter[example['conn']] += 1
    print(conn_counter)

    rels = ['N/A']
    rel2id = {rel: idx for idx, rel in enumerate(rels)}
    conns = sorted(list(conn_counter.keys()))
    conn2id = {conn: idx for idx, conn in enumerate(conns)}

    examples = [[], [], []]
    for si, split in enumerate(['train', 'dev', 'test']):
        for example in data[split]:
            conn = example['conn']
            rel = 'N/A'
            example['conn'] = conn2id[conn]
            example['rel'] = rel2id[rel]
            examples[si].append(example)

    train_dataset = ConnData(examples[0])
    dev_dataset = ConnData(examples[1])
    test_dataset = ConnData(examples[2])

    return train_dataset, dev_dataset, test_dataset, rels, conns
