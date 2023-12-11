import torch
from torch import nn
import numpy as np
from collections import OrderedDict

class Interact_Layer(nn.Module):
    def __init__(self, embed_dim=16, n_heads=2, d=8):
        # attention 技术的自适应特征交叉
        super(Interact_Layer, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d = d
        self.W_Q = nn.Linear(self.embed_dim, self.d * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.embed_dim, self.d * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.embed_dim, self.d * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d, self.embed_dim, bias=False)
        self.activate = nn.ReLU()
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, input_q, input_k, input_v):
        residual, batch_size = input_q, input_q.size(0)
        Q = self.W_Q(input_q).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        K = self.W_K(input_k).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        V = self.W_V(input_v).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        # 做点积并缩放
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)   # [batch_size, n_heads, len_q, d_v] torch.Size([2, 2, 39, 6])
        multi_attention_output = context.transpose(1, 2).reshape(batch_size, -1, self.d * self.n_heads)
        multi_attention_output = self.fc(multi_attention_output)

        # 加入残差
        residual = input_q
        output = self.activate(multi_attention_output + residual)
        return output


class DeepSepsis(nn.Module):
    def __init__(self, feas_nunique, emb_dim=16, n_layers=2, num_hidden_units=[256,125],dropout=0.1):
        '''
        :param emb_dim:
        '''
        super(DeepSepsis, self).__init__()
        self.fea_uniques=feas_nunique
        self.n_layers = n_layers

        self.features_embed = nn.ModuleList([nn.Embedding(voc_size, emb_dim) for _,voc_size in feas_nunique.items()])
        for embedding in self.features_embed:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

        self.interact_layer = Interact_Layer()
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
        embed_map = torch.cat(sparse_kd_embed, dim=1)   # torch.Size([batch_size, feature_num, emb_dim])

        for _ in range(self.n_layers):
            embed_map = self.interact_layer(embed_map, embed_map, embed_map)

        dnn_out = embed_map.view(batch_size, -1)

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        out = self.dnn_linear(dnn_out)   # batch_size, 1
        out = self.sigmoid(out)
        return out

