import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper_functions import exponential_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = np.power(d_k, .5)  #1/sqrt(d_k)

    def forward(self, Q, K, V, mask=None):
        # Q.shape and K.shape = [B, h, L, d_k]
        # V.shape = [B, h, L, d_v]
        x = torch.matmul(Q, K.permute(0,1,3,2))  # [B, h, L, L]
        x = x/self.scale  # [B, L, L]
        if mask is not None:
            x = F.softmax(exponential_mask(tensor=x, mask=mask), dim=-1)
        else:
            x = F.softmax(x, dim=-1)  # [B, h, L, L]
        x = torch.matmul(x, V)  # [B, h, L, L] x [B, h, L, d_v] = [B, h, L, d_v]
        return x

