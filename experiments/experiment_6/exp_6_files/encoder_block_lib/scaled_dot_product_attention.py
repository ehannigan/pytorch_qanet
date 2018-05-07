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
        # Q.shape and K.shape = [B, L, d_k]
        # V.shape = [B, L, d_v]
        x = torch.bmm(Q, K.permute(0,2,1))  # [B, L, L]
        x = x/self.scale  # [B, L, L]
        if mask is not None:
            x = F.softmax(exponential_mask(tensor=x, mask=mask), dim=2)
        else:
            x = F.softmax(x, dim=2)  # [B, L, L]
        x = torch.bmm(x, V)  # [B, L, d_v]
        return x

