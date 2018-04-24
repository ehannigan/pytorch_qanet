import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from context_query_attention_lib.trilinear_similarity import TrilinearSimilarity

class ContextQueryAttention(nn.Module):
    def __init__(self, config, C_shape, Q_shape):
        # C_shape = [hidden_size, context_limit]
        super(ContextQueryAttention, self).__init__()
        self.config = config
        self.trilinear_similarity = TrilinearSimilarity(in_shape=C_shape, out_shape=1)

    def forward(self, context, question):
        # context shape = [N, hidden_size, context_limit]
        # question shape = [N, hidden_size, question_limit]
        tiled_context = context.unsqueeze(3).repeat(1,1, 1, self.config.question_limit).permute(0,2,3,1).contiguous()
        tiled_question = question.unsqueeze(2).repeat(1, 1, self.config.context_limit, 1).permute(0,2,3,1).contiguous()
        # C: tiled context (N, context_limit, question_limit, d)
        # Q: tiled question (N, context_limit, question_limit, d)
        # M: tiled_context*tiled_question (N, context_limit, question_limit, d)
        S = self.trilinear_similarity(tiled_context, tiled_question, tiled_context*tiled_question)
        #print('completed S calculation')
        # S.shape = [Batch_size, context_length, question_length]
        context_t = context.permute(0,2,1)
        question_t = question.permute(0,2,1)
        S_row_norm = F.softmax(S, dim=2) #apply softmax over all questions
        S_col_norm_T = F.softmax(S, dim=1).permute(0,2,1) #apply softmax over all contexts
        context2question = torch.bmm(S_row_norm, question_t).permute(0,2,1)
        question2context =torch.bmm(torch.bmm(S_row_norm, S_col_norm_T), context_t).permute(0,2,1)
        #print('completed q2c calc')
        attention = [context, context2question, (context*context2question).contiguous(), (context*question2context).contiguous()]
        return attention



