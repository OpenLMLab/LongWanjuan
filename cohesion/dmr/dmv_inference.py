import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import DMVModel
from utils import read_discovery_connectives, get_joint_collate_fn


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--phi_lr', type=float, default=1e-2)
    parser.add_argument('--psi_lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--EM_batch_size', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--plm_name_or_path', default='roberta-base')
    parser.add_argument('--data_path', default='discovery_base.json')
    args = parser.parse_args()
    return args


def mprint(s, fp=None):
    print(s)
    if fp is not None:
        print(s, file=fp)


def check_topk_connectives(model, conns, k=4, fp=None):
    z2c = model.get_z2c()
    for zi, prob_i in enumerate(z2c):
        values, indices = prob_i.topk(k)
        mprint(
            f'{zi:02d}' + ','.join([f'{conns[i]:>15}:{v.item():.4f}' for i, v in zip(indices, values)]),
            fp=fp    
        )


if __name__ == '__main__':
    args = get_args()
    train_dataset, dev_dataset, test_dataset, rels, conns = \
        read_discovery_connectives(args.data_path)
    model = DMVModel(args.plm_name_or_path, C=len(conns), K=args.K).to(0)
    ckpt = torch.load(f'K{args.K}_ckpt/em-conn.pt')
    print(f'loading model with test-acc {ckpt["test-acc"]}')
    model.load_state_dict(ckpt['weights'], strict=False)
    check_topk_connectives(model, conns)

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name_or_path)
    collate_fn = get_joint_collate_fn(tokenizer, 0)
    batch_size = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for batch in test_loader:
        ret = model(**batch)
        print("Log Probability of z (latent discourse):")
        print(ret['logp_z'].size())
        print("Log Probability of c (connective):")
        print(ret['logp_c'].size())
        break
