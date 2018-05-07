import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableCnn(nn.Module):
    # https://www.google.com/search?q=pytorch+depthwise+then+pointwise+convolution&oq=pytorch+depthwise+then+pointwise+convolution&aqs=chrome..69i57.8734j0j4&sourceid=chrome&ie=UTF-8
    def __init__(self, C_in, C_out, kernel_size, padding):
        super(DepthwiseSeparableCnn, self).__init__()
        self.depthwise = nn.Conv1d(in_channels=C_in,
                                   out_channels=C_in,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=padding,
                                   groups=C_in)
        self.pointwise = nn.Conv1d(in_channels=C_in,
                                   out_channels=C_out,
                                   kernel_size=1,
                                   stride=1,
                                   groups=1)  # no padding because kernel=1


    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x