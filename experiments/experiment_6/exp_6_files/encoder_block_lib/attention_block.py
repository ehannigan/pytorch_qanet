import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock,self).__init__()
        self.layer_norm = nn.LayerNorm()

    def forward(self, x):
        return x