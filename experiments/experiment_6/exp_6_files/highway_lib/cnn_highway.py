import torch
from torch import nn
from highway_lib.cnn_highway_unit import CnnHighwayUnit
class CnnHighway(nn.Module):
    """Pytorch nn.Module that list of single CnnHighwayUnit

    """
    def __init__(self, num_layers, input_size, d_model, kernel_size, dropout=1, stride=1):
        super(CnnHighway, self).__init__()

        padding = self.__calc_padding(input_size, kernel_size, stride)
        self.highway_layers = nn.ModuleList([CnnHighwayUnit(input_size=input_size,
                                                            d_model=d_model,
                                                            kernel_size=kernel_size,
                                                            padding=padding,
                                                            dropout=dropout,
                                                            stride=stride) for _ in range(num_layers)])


    def forward(self, x):
        for layer in self.highway_layers:
            x = layer(x)
        return x


    def __calc_padding(self, input_size, kernel_size, stride):
        """
        we want to calculate the padding such that y.shape = x.shape for y = layer(x)
        output_height = input_height + 2*padding_height - kernel_height +1 (assuming stride=1)
        output_width = input_width + 2*padding_width - kernel_width + 1 (assuming stride=1)
        we want output_height = input_height and output_width = input_width. Therefore...
        padding_height = (kernel_height - 1)/2
        padding_width = (kernel_width - 1)/2
        """
        # default of pytorch for input_size = (C_in, H_in, W_in)
        if len(input_size) == 3:
            if stride != (1,1):
                raise ValueError("calc padding only works for stride=(1,1)")
            padding = (0,0)
            if kernel_size[0]%2 == 0 or kernel_size[1]%2 == 0:
                raise ValueError("the kernel size: {} is incompatible with CnnHighway. With this kernel, the conv output shape will not equal the input shape".format(kernel_size))
            padding_height = int((kernel_size[0] - 1)/2)
            padding_width = int((kernel_size[1] - 1)/2)
            return (padding_height, padding_width)
        if len(input_size) == 2:
            if stride != 1:
                raise ValueError("calc padding only works for stride=(1)")
            padding = int((kernel_size -1)/2)
            return padding