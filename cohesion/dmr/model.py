import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DMVModel(nn.Module):
    def __init__(self, plm_path, C=30, K=11, d=768):
        super(DMVModel, self).__init__()
        self. K = K
        self.bert = AutoModel.from_pretrained(plm_path)
        self.agg = nn.Linear(4 * d, d)
        self.h2z = torch.nn.Linear(d, K)
        self.dropout = nn.Dropout(0.1)
        self.z2c = nn.Parameter(torch.empty(K, C))
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.z2c.data.uniform_(-1. / K, 1. / K)
        
    def forward(self, u_dict, v_dict, labels=None, **kwargs):
        h = self.get_h(u_dict, v_dict)  # [B, d]
        logp_z = self.h2z(h).log_softmax(dim=-1)  # [B, K]
        logp_z2c = self.z2c.log_softmax(dim=-1)  # [K, C]
        logp_z_c = logp_z[..., None] + logp_z2c[None, ...]  # [B, K, C]
        logp_c = logp_z_c.logsumexp(dim=1)  # [B, C]

        return {'logp_z': logp_z, 'logp_c': logp_c}
    
    def train_z(self, u_dict, v_dict, labels=None, **kwargs):
        h = self.get_h(u_dict, v_dict)  # [B, d]
        
        logp_z = self.h2z(h).log_softmax(dim=-1)  # [B, K]
        # logp_z2c = self.z2c.log_softmax(dim=-1)  # [K, C]
        # logp_z_c = logp_z[..., None] + logp_z2c[None, ...]  # [B, K, C]
        # logp_c = logp_z_c.logsumexp(dim=1)  # [B, C]\

        
        # M-step
        # bsz = h.size(0)
        # log_likelihood = torch.stack([
        #     logp_z_c[i, :, labels[i]] for i in range(bsz)
        # ], dim=0)  # [B, K]
        # p_z_post = log_likelihood.softmax(dim=1).detach()  # [B, K]
        p_z_post = kwargs['p_z_post']
        
        # loss = self.kl_loss(logp_z, p_z_post)
        # equivalent cross entropy between p(z|c) and p(z)
        loss = self.ce_loss(logp_z, p_z_post)
        
        return loss
        
    def train_c(self, u_dict, v_dict, labels=None, **kwargs):
        h = self.get_h(u_dict, v_dict)  # [B, d]
        
        logp_z = self.h2z(h).log_softmax(dim=-1)  # [B, K]
        logp_z2c = self.z2c.log_softmax(dim=-1)  # [K, C]
        logp_z_c = logp_z[..., None] + logp_z2c[None, ...]  # [B, K, C]
        logp_c = logp_z_c.logsumexp(dim=1)  # [B, C]
        
        # cross entropy between p(c|z) and gt
        loss = self.ce_loss(logp_c, labels)
        return loss

    @torch.no_grad()
    def E_step(self, u_dict, v_dict, labels=None, **kwargs):
        h = self.get_h(u_dict, v_dict)
        
        # E-step
        logp_z = self.h2z(h).log_softmax(dim=-1)  # [B, K]
        logp_z2c = self.z2c.log_softmax(dim=-1)  # [K, C]
        logp_z_c = logp_z[..., None] + logp_z2c[None, ...]  # [B, K, C]
        
        bsz = h.size(0)
        log_likelihood = torch.stack([
            logp_z_c[i, :, labels[i]] for i in range(bsz)
        ], dim=0)  # [B, K]
        p_z_post = log_likelihood.softmax(dim=1)  # [B, K]
        
        return p_z_post

    @torch.no_grad()
    def inference(self, u_dict, v_dict, labels=None, **kwargs):
        h = self.get_h(u_dict, v_dict)
        
        # E-step
        logp_z = self.h2z(h).log_softmax(dim=-1)  # [B, K]
        logp_z2c = self.z2c.log_softmax(dim=-1)  # [K, C]
        logp_z_c = logp_z[..., None] + logp_z2c[None, ...]  # [B, K, C]
        logp_c = logp_z_c.logsumexp(dim=1)  # [B, C]\
        
        return {'logp_z': logp_z, 'logp_c': logp_c, 'closs': self.ce_loss(logp_c, labels)}
        
    def get_h(self, u_dict, v_dict):
        u_vec = self.bert(**u_dict)[1]  # [B, d]
        v_vec = self.bert(**v_dict)[1]  # [B, d]
        h = self.agg(torch.cat([u_vec, v_vec, u_vec-v_vec, u_vec*v_vec], dim=-1))  # [B, d]
        
        return h
    
    def get_z2c(self):
        return self.z2c.softmax(dim=-1)
