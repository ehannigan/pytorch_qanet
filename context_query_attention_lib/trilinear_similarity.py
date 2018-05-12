import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class TrilinearSimilarity(nn.Module):
    def __init__(self, in_shape, out_shape=1, dropout=0):
        super(TrilinearSimilarity, self).__init__()
        # in_shape = [hidden_size, context or question limit]
        self.out_shape = out_shape
        self.fc = nn.Linear(in_features=in_shape[0]*3, out_features=self.out_shape)
        self.trilinear_dropout = nn.Dropout(dropout)

    def forward(self, C, Q, M):
        # C: tiled context (N, context_limit, question_limit, d)
        # Q: tiled question (N, context_limit, question_limit, d)
        # M: tiled_context*tiled_question (N, context_limit, question_limit, d)
        original_shape = C.shape
        flat_C = self.__flatten(C)  # (N*CL*QL, d)
        flat_Q = self.__flatten(Q)  # (N*CL*QL, d)
        flat_M = self.__flatten(M)  # (N*CL*QL, d)
        S_flat = self.fc(torch.cat([flat_C, flat_Q, flat_M], dim=1))
        # S_flat.shape = (N*CL*QL, 1)
        S = self.__unflatten(S_flat, original_shape).squeeze(-1)
        # S.shape = (N, CL, QL, 1).squeeze(-1) = (N, CL, QL)
        S = self.trilinear_dropout(S)
        return S


    def __flatten(self, x):
        original_shape = x.shape
        flat = x.view(original_shape[0]*original_shape[1]*original_shape[2], original_shape[3])
        return flat

    def __unflatten(self, S_flat, original_shape):
        S = S_flat.view(original_shape[0], original_shape[1], original_shape[2], self.out_shape)
        return S