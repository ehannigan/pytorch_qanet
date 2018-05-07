import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEmb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvEmb, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        batchsize = x.shape[0]
        context_or_question_limit = x.shape[1]
        glove_char_dim = x.shape[2]
        char_limit = x.shape[3]
        x = F.relu(self.conv(x.view(batchsize * context_or_question_limit, glove_char_dim, char_limit)))
        return x