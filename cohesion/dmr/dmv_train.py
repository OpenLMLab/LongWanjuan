import os
import torch
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from utils import read_discovery_connectives, get_joint_collate_fn
from model import DMVModel


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--phi_lr', type=float, default=1e-2)
    parser.add_argument('--psi_lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--EM_batch_size', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--plm_name_or_path', default='roberta')
    parser.add_argument('--data_path', default='/mnt/inspurfs/lvkai/llm_data/dmv_wanjuan/merged_data.json')
    args = parser.parse_args()
    return args


def mprint(s, fp=None):
    print(s)
    if fp is not None:
        print(s, file=fp)


@torch.no_grad()
def eval(model, loader, label_names, split='dev', max_k =10):
    model.eval()
    num_class = len(label_names)
    confusion_matrix = torch.zeros(num_class, num_class).int()
    acc_topk = torch.zeros(max_k)
    for batch in tqdm(loader):
        ret = model.inference(**batch)
        predicted = ret['logp_c'].argmax(dim=1)
        labels = batch['labels']

        for k in range(1, max_k+1):
            topk = ret['logp_c'].topk(k, dim=1).indices
            acc_topk[k-1] += torch.any(topk == labels[..., None], dim=1).sum().item()

        for pred, label in zip(predicted, labels):
            confusion_matrix[pred, label] += 1

    total = confusion_matrix.sum()
    correct = confusion_matrix[torch.eye(num_class).bool()].sum()
    acc = correct / total

    correct = confusion_matrix[torch.eye(num_class).bool()]
    sum1 = confusion_matrix.sum(dim=0)
    sum2 = confusion_matrix.sum(dim=1)
    f1 = 2 * correct / (sum1 + sum2)
    f1[f1.isnan()] = 0.

    out_dict = {f'{split}-acc': acc, f'{split}-macro-f1': f1.mean(), f'{split}-acc-topk': acc_topk/total}

    model.train()

    return out_dict

def check_topk_connectives(model, conns, k=4, fp=None):
    z2c = model.get_z2c()
    for zi, prob_i in enumerate(z2c):
        values, indices = prob_i.topk(k)
        mprint(
            f'{zi:02d}' + ','.join([f'{conns[i]:>15}:{v.item():.4f}' for i, v in zip(indices, values)]),
            fp=fp    
        )


def main():
    args = get_args()
    train_dataset, dev_dataset, test_dataset, rels, conns = \
        read_discovery_connectives(args.data_path)
    print(len(train_dataset) + len(dev_dataset) + len(test_dataset))
    print(len(train_dataset))
    print(len(rels), len(conns))

    save_dir = (
        f"./save_zh/K_{args.K}_EM_{args.EM_batch_size}_bsz_"
        f"{args.batch_size}_psi_{args.psi_lr}_phi_{args.phi_lr}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name_or_path)
    collate_fn = get_joint_collate_fn(tokenizer, 0)
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    ckpt_path = '/cpfs01/shared/public/dmv_ckpts/em-conn.pt'
    model = DMVModel(args.plm_name_or_path, C=len(conns), K=args.K).to(0)
    ckpt = torch.load(ckpt_path)
    print(f'loading model with test-acc {ckpt["test-acc"]}')
    model.load_state_dict(ckpt['weights'], strict=False)
    z_optimizer = AdamW(
        list(model.bert.parameters()) +\
        list(model.agg.parameters()) +\
        list(model.h2z.parameters()),
        args.psi_lr
    )
    c_optimizer = AdamW([model.z2c], args.phi_lr)

    total_steps = len(train_loader) * args.num_epochs
    # warmup_steps = int(total_steps * 0.06)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    logfile = os.path.join(save_dir, 'log')
    open(logfile, 'w').close()
    step = 0
    best_score = 0.
    for ei in range(args.num_epochs):
        batches = []
        for bi, batch in tqdm(enumerate(train_loader)):
            batches.append(batch)
            step += 1
            
            if step % (args.eval_steps * args.EM_batch_size) == 0:
                # E-step
                for batch in tqdm(batches):
                    p_z_post = model.E_step(**batch)
                    batch['p_z_post'] = p_z_post
                
                for batch in tqdm(batches):
                    loss = model.train_z(**batch)
                    z_optimizer.zero_grad()
                    loss.backward()
                    z_optimizer.step()

                for batch in tqdm(batches):
                    loss = model.train_c(**batch)
                    c_optimizer.zero_grad()
                    loss.backward()
                    c_optimizer.step()
                
                batches = []
                dev_ret = eval(
                    model, dev_loader, conns,
                )
                test_ret = eval(
                    model, test_loader, conns, split='test'
                )
                with open(logfile, 'a') as fp:
                    mprint(f'step: {step}', fp=fp)
                    mprint(dev_ret, fp=fp)
                    mprint(test_ret, fp=fp)
                    check_topk_connectives(model, conns, fp=fp)
                    if test_ret['test-acc'] > best_score:
                        mprint('get better results!', fp)
                        best_score = test_ret['test-acc']
                        test_ret['conns'] = conns
                        test_ret['weights'] = model.state_dict()
                        torch.save(test_ret, os.path.join(save_dir, f'em-conn.pt'))
                        tokenizer.save_pretrained(os.path.join(save_dir, f'em-roberta'))
                        model.bert.save_pretrained(os.path.join(save_dir, f'em-roberta'))


if __name__ == '__main__':
    main()
