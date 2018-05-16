import torch
from torch import nn
from highway_lib.cnn_highway_unit import CnnHighwayUnit
from helper_functions import calc_padding
class CnnHighway(nn.Module):
    """Pytorch nn.Module that list of single CnnHighwayUnit

    """
    def __init__(self, num_layers, input_shape, out_channels, kernel_size, dropout=0, stride=1):
        super(CnnHighway, self).__init__()

        padding = calc_padding(input_shape, kernel_size, stride)
        self.highway_layers = nn.ModuleList([CnnHighwayUnit(input_shape=input_shape,
                                                            out_channels=out_channels,
                                                            kernel_size=kernel_size,
                                                            padding=padding,
                                                            dropout=dropout,
                                                            stride=stride) for _ in range(num_layers)])


    def forward(self, x):
        for layer in self.highway_layers:
            x = layer(x)
        return x


