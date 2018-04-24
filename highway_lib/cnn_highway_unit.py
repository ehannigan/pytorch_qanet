import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#https://arxiv.org/pdf/1507.06228.pdf
#https://github.com/c0nn3r/pytorch_highway_networks/blob/master/layers/highway.py
class CnnHighwayUnit(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, padding, stride=(1,1)):
        super(CnnHighwayUnit, self).__init__()
        # default of pytorch for input_size = (C_in, H_in, W_in)

        #print('input size', input_size)
        self.conv_h = nn.Conv1d(in_channels=input_size[0], out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_t = nn.Conv1d(in_channels=input_size[0], out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        H = self.conv_h(x)
        H = F.relu(H)
        T = F.sigmoid(self.conv_t(x))
        matrix_1 = torch.ones_like(T)

        #y = torch.add(torch.mul(H,T), torch.mul(x, torch.add(matrix_1, -T)))
        y = H * T + x * (matrix_1 - T)
        return y


