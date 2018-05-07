import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import LayerDropout

from encoder_block_lib.forward_block import ForwardBlock
from encoder_block_lib.multi_head_attention_block import  MultiHeadAttentionBlock
from encoder_block_lib.repeated_cnn_block import RepeatedCnnBlock
class EncoderBlock(nn.Module):
    def __init__(self, num_conv_blocks, num_heads, d_model, kernel_size, input_shape, depthwise, norm, layer_dropout, general_dropout, total_layers):
        super(EncoderBlock, self).__init__()
        # depthwise convolution

        self.conv_block = RepeatedCnnBlock(num_conv_blocks=num_conv_blocks,
                                           d_model=d_model,
                                           kernel_size=kernel_size,
                                           input_shape=input_shape,
                                           depthwise=depthwise,
                                           norm=norm,
                                           layer_dropout=layer_dropout,
                                           total_layers=total_layers)
        self.multi_head_attention_block = MultiHeadAttentionBlock(num_heads=num_heads,
                                                                  d_model=d_model,
                                                                  norm=norm)
        #self.multi_head_dropout = LayerDropout(layer_dropout=layer_dropout, total_layers=total_layers)
        #attention_input_shape = self.__get_attention_input_shape(input_shape)
        #self.attention_block = AttentionBlock(input_shape)
        #linear_input_shape = self.__get_linear_input_shape(input_shape)
        self.forward_block = ForwardBlock(input_shape=input_shape, d_model=d_model, norm=norm)
        #self.forward_dropout = LayerDropout(layer_dropout=layer_dropout, total_layers=total_layers)


    def forward(self, x, mask=None):
        x = self.conv_block(x)
        x = self.multi_head_attention_block(x, mask=mask)
        #x = self.multi_head_dropout(x)
        x = self.forward_block(x)
        #x = self.forward_dropout(x)
        return x

