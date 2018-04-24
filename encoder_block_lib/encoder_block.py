import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from encoder_block_lib.cnn_block import CnnBlock
from encoder_block_lib.linear_block import LinearBlock
from encoder_block_lib.attention_block import AttentionBlock

class EncoderBlock(nn.Module):
    def __init__(self, num_conv_layers, hidden_size, kernel_size, input_shape, depthwise):
        super(EncoderBlock, self).__init__()
        # depthwise convolution
        #self.position_encoding =
        self.conv_block = CnnBlock(num_conv_layers, hidden_size, kernel_size, input_shape, depthwise=depthwise)
        #attention_input_shape = self.__get_attention_input_shape(input_shape)
        #self.attention_block = AttentionBlock(input_shape)
        #linear_input_shape = self.__get_linear_input_shape(input_shape)
        self.linear_block = LinearBlock(input_shape, hidden_size)


    def forward(self, x):
        x = self.conv_block(x)
        #x = self.attention_block(x)
        x = self.linear_block(x)
        return x

    # def get_positional_encoding(self, input_shape, max_timescale=1.0, min_timescale=1.0e-4):
    #     # https://github.com/minsangkim142/QANet/blob/6540d9ad68f8a713b58f772d0ea5d4a8ff8eb27b/layers.py#L91-L109
    #     # input shape = (hidden_size, context/question_limit)
    #     position = torch.range(input_shape[1])
    #     num_timescales = input_shape[0]//2 
    #     log_timescale_increment = (math.log(float(max_timescale)/float(min_timescale))/(num_timescales-1))


    #     signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    #     signal = torch.pad(signal, [[0,0], [0.torch.mod(channels,2)]])


    # def __get_attention_input_shape(self, input_shape):
    #     test_var = torch.zeros((5,)+input_shape)
    #     x = self.conv_block(test_var)
    #     return x.shape[1:]
    #
    # def __get_linear_input_shape(self, input_shape):
    #     test_var = torch.zeros((5,) + input_shape)
    #     x = self.conv_block(test_var)
    #     x = self.attention_block(x)
    #     return x.shape[1:]