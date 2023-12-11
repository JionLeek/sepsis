import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class LogisticRegression(nn.Module):
    """
        LR
    """
    def __init__(self, feas_nunique):
        '''
        :param emb_dim:
        '''
        super(LogisticRegression, self).__init__()
        self.fea_uniques=feas_nunique

        self.features_embed = nn.ModuleList([nn.Embedding(voc_size, 1) for _,voc_size in feas_nunique.items()])
        for embedding in self.features_embed:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
        self.output = nn.Linear(len(feas_nunique), 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,))) 
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        batch_size = x.size(0)
        sparse_kd_embed = [emb(x[:, i].unsqueeze(1)) for i, emb in enumerate(self.features_embed)]
        embed_map = torch.cat(sparse_kd_embed, dim=1)  
        embed_map = embed_map.view(batch_size, -1)

        out = self.output(embed_map) + self.bias
        out = self.sigmoid(out)
        return out
