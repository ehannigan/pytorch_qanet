import torch
from torch.autograd import Variable
import torch.nn as nn
from encoder_block_lib.scaled_dot_product_attention import ScaledDotProductAttention
from torch.nn import init
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.d_v = self.d_k
        self.W_q = nn.Parameter(torch.zeros([num_heads, d_model, self.d_k]))
        self.W_k = nn.Parameter(torch.zeros([num_heads, d_model, self.d_k]))
        self.W_v = nn.Parameter(torch.zeros([num_heads, d_model, self.d_v]))
        self.W_o = nn.Parameter(torch.zeros([num_heads*self.d_v, d_model]))

        #self.dropout = nn.Dropout()
        init.xavier_uniform(self.W_q.data)
        init.xavier_uniform(self.W_k.data)
        init.xavier_uniform(self.W_v.data)
        init.xavier_uniform(self.W_o.data)

        self.attention = ScaledDotProductAttention(d_k=self.d_k, d_v=self.d_v, d_model=d_model)

    def forward(self, Q, K, V, mask=None):
        heads = []
        batchsize = Q.shape[0]
        for h in range(self.num_heads):
            q = torch.bmm(Q.permute(0, 2, 1), self.W_q[h].unsqueeze(0).expand(batchsize, self.d_model, self.d_k))  #[B, L, d_k] = [B, L, d_model] * [_, d_model, d_k]
            k = torch.bmm(K.permute(0, 2, 1), self.W_k[h].unsqueeze(0).expand(batchsize, self.d_model, self.d_k))  #[B, L, d_k]
            v = torch.bmm(V.permute(0, 2, 1), self.W_v[h].unsqueeze(0).expand(batchsize, self.d_model, self.d_v))  #[B, L, d_v]
            heads.append(self.attention(q, k, v, mask=mask))
        heads_concat = torch.cat(heads, dim=2)  # [B, L, dv*h]
        attend = torch.bmm(heads_concat, self.W_o.unsqueeze(0).expand(batchsize, self.num_heads*self.d_v, self.d_model))  #[B, L, d_model]
        return attend



