import torch
from torch.autograd import Variable
import torch.nn as nn
from encoder_block_lib.scaled_dot_product_attention import ScaledDotProductAttention
from torch.nn import init
import copy
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.d_v = self.d_k
        #method3
        self.Ws = nn.ModuleList([copy.deepcopy(nn.Linear(in_features=d_model, out_features=d_model)) for _ in range(4)])

        #method2
        # self.W_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        # self.W_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        # self.W_v = nn.Parameter(torch.FloatTensor(d_model, d_model))
        # self.W_o = nn.Parameter(torch.FloatTensor(num_heads*self.d_v, d_model))

        # init.xavier_uniform(self.W_q.data)
        # init.xavier_uniform(self.W_k.data)
        # init.xavier_uniform(self.W_v.data)
        # init.xavier_uniform(self.W_o.data)

        #method1
        # self.W_q = nn.Parameter(torch.zeros(num_heads, d_model, d_k))
        # self.W_k = nn.Parameter(torch.FloatTensor(num_heads, d_model, d_k))
        # self.W_v = nn.Parameter(torch.FloatTensor(num_heads, d_model, d_v))
        # self.W_o = nn.Parameter(torch.FloatTensor(num_heads*self.d_v, d_model))

        self.attention = ScaledDotProductAttention(d_k=self.d_k, d_v=self.d_v, d_model=d_model)

    def forward(self, Q, K, V, mask=None):
        heads = []
        # Q = [B, d_model, CL]
        batchsize = Q.shape[0]
        #method3
        q, k, v = [self.W(x.permute(0,2,1)).view(batchsize, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3) for W, x in zip(self.Ws, (Q,K,V))]
        heads_concat = self.attention(Q=q, K=k, V=v, mask=mask).permute(0, 2, 1, 3).contiguous().view(batchsize, -1, self.d_model)
        attend = self.Ws[-1](heads_concat)

        #method2 still memory issues
        # Q_h = torch.bmm(Q.permute(0,2,1), self.W_q.unsqueeze(0).expand(batchsize, self.d_model, self.d_model)).contiguous().view(batchsize, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # [B, CL, d_model]x[B, d_model, d_model] = [B, CL, d_model]
        # K_h = torch.bmm(K.permute(0,2,1), self.W_k.unsqueeze(0).expand(batchsize, self.d_model, self.d_model)).contiguous().view(batchsize, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        # V_h = torch.bmm(V.permute(0,2,1), self.W_v.unsqueeze(0).expand(batchsize, self.d_model, self.d_model)).contiguous().view(batchsize, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        # heads_concat = self.attention(Q=Q_h, K=K_h, V=V_h, mask=mask).permute(0, 2, 1, 3).contiguous().view(batchsize, -1, self.d_model)  # [B, h, L, d_v] => [B, L, h, d_v] => [B, L, d_model, L]


        #method1 memory issues
        # for h in range(self.num_heads):
        #     q = torch.bmm(Q.permute(0, 2, 1), self.W_q[h].unsqueeze(0).expand(batchsize, self.d_model, self.d_k))  #[B, L, d_k] = [B, L, d_model] * [_, d_model, d_k]
        #     k = torch.bmm(K.permute(0, 2, 1), self.W_k[h].unsqueeze(0).expand(batchsize, self.d_model, self.d_k))  #[B, L, d_k]
        #     v = torch.bmm(V.permute(0, 2, 1), self.W_v[h].unsqueeze(0).expand(batchsize, self.d_model, self.d_v))  #[B, L, d_v]
        #     heads.append(self.attention(q, k, v, mask=mask))
        # heads_concat = torch.cat(heads, dim=2)  # [B, L, dv*h]
        #attend = torch.bmm(heads_concat, self.W_o.unsqueeze(0).expand(batchsize, self.num_heads*self.d_v, self.d_model))  #[B, L, d_model]x[B, d_model, d_model] = [B, L, d_mode]


        return attend



