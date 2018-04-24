import torch.nn as nn
import torch.nn.functional as F
from encoder_block_lib.depthwise_separable_cnn import DepthwiseSeparableCnn

class CnnBlock(nn.Module):
    def __init__(self, num_conv_layers, num_filters, kernel_size, input_shape, depthwise=False):
        super(CnnBlock, self).__init__()
        self.layer_norms = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # input to each conv layer must have the same shape so padding must be added to make sure input shape = output shape
        padding = self.__calc_padding(input_shape, kernel_size)
        for i in range(num_conv_layers):
            #print('adding conv layer i', i)
            # no layer norm yet in pytorch .3
            self.layer_norms.append(nn.BatchNorm1d(input_shape[0]))
            if depthwise:
                self.conv_layers.append(DepthwiseSeparableCnn(C_in=num_filters, C_out=num_filters, kernel_size=kernel_size, padding=padding)) #stride automatically=1
            else:
                self.conv_layers.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=padding))


    def forward(self, x):
        #print('cnn block forward x.size', x.shape)
        for i in range(len(self.conv_layers)):
            x = self.layer_norms[i](x)
            x = F.relu(self.conv_layers[i](x))
        return x

    def __calc_padding(self, input_shape, kernel_size, stride=1):
        """
        we want to calculate the padding such that y.shape = x.shape for y = layer(x)
        output_height = input_height + 2*padding_height - kernel_height +1 (assuming stride=1)
        output_width = input_width + 2*padding_width - kernel_width + 1 (assuming stride=1)
        we want output_height = input_height and output_width = input_width. Therefore...
        padding_height = (kernel_height - 1)/2
        padding_width = (kernel_width - 1)/2
        """
        # default of pytorch for input_size = (C_in, H_in, W_in)
        if len(input_shape) == 3:
            if stride != (1,1):
                raise ValueError("calc padding only works for stride=(1,1)")
            padding = (0,0)
            if kernel_size[0]%2 == 0 or kernel_size[1]%2 == 0:
                raise ValueError("the kernel size: {} is incompatible with CnnHighway. With this kernel, the conv output shape will not equal the input shape".format(kernel_size))
            padding_height = int((kernel_size[0] - 1)/2)
            padding_width = int((kernel_size[1] - 1)/2)
            return (padding_height, padding_width)
        if len(input_shape) == 2:
            if stride != 1:
                raise ValueError("calc padding only works for stride=(1)")
            padding = int((kernel_size -1)/2)
            return padding