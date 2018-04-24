import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class LinearBlock(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(LinearBlock, self).__init__()
        #print('creating linear block')
        # no layernorm in pytorch .3
        self.layer_norm = nn.BatchNorm1d(hidden_size)
        total_features = input_shape[0]*input_shape[1]
        #print('total features', total_features)
        self.fc = nn.Linear(in_features=total_features, out_features=total_features)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x_flat = self.__flatten(x)
        x_flat = self.fc(x_flat)
        x = self.__unflatten(x_flat, residual.shape)
        return x + residual


    def __flatten(self, x):
        batch_size = x.shape[0]
        feature_size = x.shape[1]*x.shape[2]
        flat = x.view(batch_size, feature_size)
        return flat


    def __unflatten(self, x_flat, original_shape):
        x = x_flat.view(original_shape)
        return x