import torch
import torch.nn as nn
from torch.nn import init

class LinearLogit(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLogit, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros(1, in_features))
        init.xavier_uniform(self.W.data)
    def forward(self, x):
        batchsize = x.shape[0]
        w0_dim = self.W.shape[0]
        w1_dim = self.W.shape[1]
        x = torch.bmm(self.W.unsqueeze(0).expand(batchsize, w0_dim, w1_dim), x)
        x = x.squeeze()
        return x