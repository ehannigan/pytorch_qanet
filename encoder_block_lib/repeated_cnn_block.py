import torch.nn as nn
import torch.nn.functional as F
from encoder_block_lib.depthwise_separable_cnn import DepthwiseSeparableCnn
from encoder_block_lib.cnn_block import CNNBlock
from helper_functions import LayerDropout
from helper_functions import calc_padding

class RepeatedCnnBlock(nn.Module):
    def __init__(self, num_conv_blocks, out_channels, kernel_size, input_shape, depthwise, norm, layer_dropout, total_layers):
        super(RepeatedCnnBlock, self).__init__()
        self.cnn_block_list = nn.ModuleList()
        #self.cnn_block_dropout_list = nn.ModuleList()
        # input to each conv layer must have the same shape so padding must be added to make sure input shape = output shape
        padding = calc_padding(input_shape, kernel_size)
        for i in range(num_conv_blocks):
            self.cnn_block_list.append(CNNBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, depthwise=depthwise, norm=norm))
            #self.cnn_block_dropout_list.append(LayerDropout(total_layers=total_layers, layer_dropout=layer_dropout))

    def forward(self, x):
        #print('cnn block forward x.size', x.shape)
        for i, cnn_block in enumerate(self.cnn_block_list):
            x = cnn_block(x)
            #x = self.cnn_block_dropout_list[i](x)
        return x
