import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from context_query_attention_lib.trilinear_similarity import TrilinearSimilarity
from helper_functions import exponential_mask


class ContextQueryAttention(nn.Module):
    def __init__(self, config, C_shape, Q_shape, dropout=0):
        # C_shape = [hidden_size, context_limit]
        super(ContextQueryAttention, self).__init__()
        self.config = config
        self.trilinear_similarity = TrilinearSimilarity(in_shape=C_shape, out_shape=1, dropout=dropout)

    def forward(self, context, question, context_word_mask=None, question_word_mask=None):
        # context shape = [N, hidden_size, context_limit]
        # question shape = [N, hidden_size, question_limit]
        batchsize = context.shape[0]
        hidden_size = context.shape[1]
        context_limit = context.shape[2]
        question_limit = question.shape[2]

        tiled_context = context.unsqueeze(3).expand(batchsize, hidden_size, context_limit, question_limit).permute(0, 2, 3, 1).contiguous()
        tiled_question = question.unsqueeze(2).expand(batchsize, hidden_size, context_limit, question_limit).permute(0, 2, 3, 1).contiguous()
        # you can also use repeat, but I realized that expand allows you to broadcast in the same way but you save memory
        #tiled_context = context.unsqueeze(3).repeat(1,1, 1, self.config.question_limit).permute(0,2,3,1).contiguous()
        #tiled_question = question.unsqueeze(2).repeat(1, 1, self.config.context_limit, 1).permute(0,2,3,1).contiguous()
        # C: tiled context (N, context_limit, question_limit, d)
        # Q: tiled question (N, context_limit, question_limit, d)
        # M: tiled_context*tiled_question (N, context_limit, question_limit, d)

        S = self.trilinear_similarity(tiled_context, tiled_question, tiled_context*tiled_question)
        # S.shape = [Batch_size, context_length, question_length]
        context_t = context.permute(0,2,1)
        question_t = question.permute(0,2,1)
        if self.config.c2q_mask:
            S_row_norm =  F.softmax(exponential_mask(tensor=S, mask=question_word_mask.unsqueeze(1).expand(batchsize, context_limit, question_limit)),dim=2) #apply softmax over all questions
            S_col_norm_T = F.softmax(exponential_mask(tensor=S, mask=context_word_mask.unsqueeze(2).expand(batchsize, context_limit, question_limit)), dim=1).permute(0,2,1)  #apply softmax over all contexts
        else:
            S_row_norm = F.softmax(S, dim=2) #apply softmax over all questions
            S_col_norm_T = F.softmax(S, dim=1).permute(0,2,1) #apply softmax over all contex
        context2question = torch.bmm(S_row_norm, question_t).permute(0,2,1)
        question2context =torch.bmm(torch.bmm(S_row_norm, S_col_norm_T), context_t).permute(0,2,1)
        #print('completed q2c calc')
        attention = [context, context2question, (context*context2question).contiguous(), (context*question2context).contiguous()]
        return attention




### code to prove to yourself that .unsqueeze.expand words the same as .unsqueeze.repeat
    # context shape = [N, hidden_size, context_limit]
    # question shape = [N, hidden_size, question_limit]

    # batchsize = 2
    # hidden_size = 3
    # context_limit = 10
    # question_limit = 8
    #
    # ts = TrilinearSimilarity(in_shape=(hidden_size, context_limit))
    #
    # context = Variable(torch.rand(batchsize, hidden_size, context_limit))
    # question = Variable(torch.rand(batchsize, hidden_size, question_limit))
    #
    # context_word_mask = Variable(torch.zeros(batchsize, context_limit))
    # question_word_mask = Variable(torch.zeros(batchsize, question_limit))
    #
    # context_word_mask[0,:3] = 1
    # question_word_mask[0, :6] = 1
    #
    # context_word_mask[1,:1] = 1
    # question_word_mask[1, :2] = 1
    #
    #
    # tiled_context1 = context.unsqueeze(3).repeat(1, 1, 1, question_limit).permute(0, 2, 3, 1).contiguous()
    # tiled_question1 = question.unsqueeze(2).repeat(1, 1, context_limit, 1).permute(0, 2, 3, 1).contiguous()
    # C1 = tiled_context1
    # Q1 = tiled_question1
    # M1 = tiled_context1 * tiled_question1
    #
    # S_try1 = ts(C1, Q1, M1)
    # print('S_try1', S_try1)
    #
    #
    # tiled_context2 = context.unsqueeze(3).expand(batchsize, hidden_size, context_limit, question_limit).permute(0, 2, 3, 1).contiguous()
    # tiled_question2 = question.unsqueeze(2).expand(batchsize, hidden_size, context_limit, question_limit).permute(0, 2, 3, 1).contiguous()
    #
    # C2 = tiled_context2
    # Q2 = tiled_question2
    # M2 = tiled_context2 * tiled_question2
    #
    # S_try2 = ts(C2, Q2, M2)
    # print('S_try2', S_try2)
    #
    # s_same = torch.eq(S_try1, S_try2)
    # print('are both s same?', s_same)
    #
    # S1_row_norm = F.softmax(exponential_mask(tensor=S_try1, mask=question_word_mask.unsqueeze(1).expand(batchsize, context_limit, question_limit)),dim=2)  # apply softmax over all questions
    # S1_col_norm_T = F.softmax(exponential_mask(tensor=S_try1, mask=context_word_mask.unsqueeze(2).expand(batchsize, context_limit, question_limit)), dim=1).permute(0,
    #                                                                                                                 2,
    #                                                                                                                 1)  # apply softmax over all contexts
    # print('s1_row_norm', S1_row_norm)
    # print('s1_col_norm', S1_col_norm_T)
    #
    # S2_row_norm = F.softmax(exponential_mask(tensor=S_try2, mask=question_word_mask.unsqueeze(1).repeat(1, context_limit, 1)), dim=2) #apply softmax over all questions
    # S2_col_norm_T = F.softmax(exponential_mask(tensor=S_try2, mask=context_word_mask.unsqueeze(2).repeat(1, 1, question_limit)), dim=1).permute(0,2,1) #apply softmax over all contexts
    # print('s2_row_norm', S2_row_norm)
    # print('s2_col_norm', S2_col_norm_T)
    #
    # row_norm_same = torch.eq(S1_row_norm, S2_row_norm)
    # col_norm_same = torch.eq(S1_col_norm_T, S2_col_norm_T)
    #
    # print('row norm same', row_norm_same)
    # print('col norm same', col_norm_same)


