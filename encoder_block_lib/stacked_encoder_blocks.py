from encoder_block_lib.encoder_block import EncoderBlock
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class StackedEncoderBlock(nn.Module):
    def __init__(self, num_encoder_blocks, num_conv_layers, hidden_size, kernel_size, input_shape, depthwise=False):
        super(StackedEncoderBlock, self).__init__()
        self.input_shape = input_shape
        self.num_encoder_blocks = num_encoder_blocks
        self.num_conv_layers = num_conv_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.map_flag = False
        self.__create_encoder_blocks()

    def __create_encoder_blocks(self):
        # print('creating encoder blocks', self.num_encoder_blocks)
        # print('num_conv_layres', self.num_conv_layers)
        # print('self.hidden_size', self.hidden_size)
        # print('self.kernel_size', self.kernel_size)
        # print('self.input_shape', self.input_shape)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(num_conv_layers=self.num_conv_layers,
                                                          hidden_size=self.hidden_size,
                                                          kernel_size=self.kernel_size,
                                                          input_shape=self.input_shape,
                                                          depthwise=self.depthwise)
                                             for _ in range(self.num_encoder_blocks)])

    def forward(self, x):
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
        return x