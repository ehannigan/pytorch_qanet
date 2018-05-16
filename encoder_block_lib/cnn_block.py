from encoder_block_lib.depthwise_separable_cnn import DepthwiseSeparableCnn
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, depthwise, norm):
        super(CNNBlock, self).__init__()
        if norm == 'batch':
            # num_features – C from an expected input of size (N,C,L) or L from input of size (N,L)
            self.norm = nn.BatchNorm1d(num_features=out_channels)
        elif norm == 'layer_norm':
            raise ValueError('no layer norm right now')

        if depthwise:
            self.cnn = DepthwiseSeparableCnn(C_in=in_channels, C_out=out_channels, kernel_size=kernel_size, padding=padding)  # stride automatically=1
        else:
            self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        residual = x.clone()
        x = self.norm(x)
        x = F.relu(self.cnn(x))
        return x + residual
