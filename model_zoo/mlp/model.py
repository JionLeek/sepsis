import torch
from torch import nn
import numpy as np
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, feas_nunique, emb_dim=16, num_hidden_units=[256,125],dropout=0.1):
        '''
        :param emb_dim:
        '''
        super(MLP, self).__init__()
        self.fea_uniques=feas_nunique

        self.features_embed = nn.ModuleList([nn.Embedding(voc_size, emb_dim) for _,voc_size in feas_nunique.items()])
        for embedding in self.features_embed:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
        
        self.all_dims = [len(feas_nunique) * emb_dim] + num_hidden_units
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout))

        self.dnn_linear = nn.Linear(self.all_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: sparse_feature [batch_size, feature_num]
        """
        batch_size = x.size(0)
        sparse_kd_embed = [emb(x[:, i].unsqueeze(1)) for i, emb in enumerate(self.features_embed)]
        embed_map = torch.cat(sparse_kd_embed, dim=1)   

        dnn_out = embed_map.view(batch_size, -1)

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        out = self.dnn_linear(dnn_out)   # batch_size, 1
        out = self.sigmoid(out)
        return out

