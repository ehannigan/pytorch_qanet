import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ForwardBlock(nn.Module):
    def __init__(self, input_shape, d_model, norm):
        super(ForwardBlock, self).__init__()
        self.d_model = d_model
        #print('creating linear block')
        # no layernorm in pytorch .3
        if norm == 'batch':
            # num_features – C from an expected input of size (N,C,L) or L from input of size (N,L)
            # num features = d_model if we apply norm before flatten. If we apply after flatten, it has to be total_features
            self.norm = nn.BatchNorm1d(d_model)
        elif norm == 'layer':
            raise ValueError('no layer norm implemented yet')
        #print('total features', total_features)
        self.W_f = torch.nn.Parameter(torch.zeros((d_model, d_model)))
        # if cuda_flag:
        #     self.W_f = self.W_f.cuda()

        init.xavier_uniform(self.W_f.data)

    def forward(self, x):
        batchsize = x.shape[0]
        residual = x.clone()
        w_f = self.W_f.unsqueeze(0).expand(batchsize, self.d_model, self.d_model)
        x = self.norm(x)
        x = torch.bmm(x.permute(0, 2, 1), w_f)
        x = F.relu(x.permute(0, 2, 1))
        return x + residual


