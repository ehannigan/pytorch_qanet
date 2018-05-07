from encoder_block_lib.encoder_block import EncoderBlock
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import LayerDropout


class StackedEncoderBlock(nn.Module):
    def __init__(self, num_encoder_blocks, num_conv_blocks, num_heads, d_model, kernel_size, input_shape, layer_dropout=1, general_dropout=1, depthwise=False, norm='batch'):
        super(StackedEncoderBlock, self).__init__()
        self.input_shape = input_shape
        self.num_encoder_blocks = num_encoder_blocks
        self.num_conv_blocks = num_conv_blocks
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.num_heads = num_heads
        self.norm = norm
        self.map_flag = False
        total_enc_layers = num_encoder_blocks * (num_conv_blocks + 2)
        self.__create_encoder_blocks(total_enc_layers=total_enc_layers, layer_dropout=layer_dropout, general_dropout=general_dropout)



    def __create_encoder_blocks(self, total_enc_layers, layer_dropout=1, general_dropout=1):
        # print('creating encoder blocks', self.num_encoder_blocks)
        # print('num_conv_layres', self.num_conv_layers)
        # print('self.hidden_size', self.hidden_size)
        # print('self.kernel_size', self.kernel_size)
        # print('self.input_shape', self.input_shape)
        self.encoder_blocks = nn.ModuleList()
        #self.encoder_block_dropouts = nn.ModuleList()
        for i in range(self.num_encoder_blocks):
            self.encoder_blocks.append(EncoderBlock(num_conv_blocks=self.num_conv_blocks,
                                                      d_model=self.d_model,
                                                      kernel_size=self.kernel_size,
                                                      input_shape=self.input_shape,
                                                      depthwise=self.depthwise,
                                                      num_heads = self.num_heads,
                                                      norm = self.norm,
                                                      layer_dropout=layer_dropout,
                                                      general_dropout=general_dropout,
                                                      total_layers=total_enc_layers))
            # if i % 2 == 0:
            #     self.encoder_block_dropouts.append(nn.Dropout(general_dropout))


    def forward(self, x, mask=None):
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x, mask)
            # if i % 2 == 0:
            #     x = self.encoder_block_dropouts[i//2](x)
        return x