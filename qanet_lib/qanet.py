from encoder_block_lib.stacked_encoder_blocks import StackedEncoderBlock
from context_query_attention_lib.context_query_attention import ContextQueryAttention
from highway_lib.cnn_highway import CnnHighway
from output_lib.linear_logit import LinearLogit
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from embed_lib.conv_emb import ConvEmb
from helper_functions import LayerDropout



class QANet(nn.Module):

    def __init__(self, config, word_emb_weights, char_emb_weights):
        super(QANet, self).__init__()

        self.encoder_masking_flag = config.self_attention_mask
        self.c2q_masking_flag = config.c2q_mask
        self.context_limit = config.context_limit
        self.question_limit = config.question_limit
        self.glove_char_dim = config.glove_char_dim
        #We additionally use dropout on word, character embeddings and between layers, where the word and character dropout rates are 0.1 and 0.05 respectively
        self.word_embedding = nn.Embedding.from_pretrained(embeddings=word_emb_weights, freeze=True)
        self.char_embedding = nn.Embedding.from_pretrained(embeddings=char_emb_weights, freeze=True)


        self.context_word_embed_dropout = nn.Dropout(config.word_emb_dropout)
        self.context_char_embed_dropout = nn.Dropout(config.char_emb_dropout)
        self.question_word_embed_dropout = nn.Dropout(config.word_emb_dropout)
        self.question_char_embed_dropout = nn.Dropout(config.char_emb_dropout)

        self.char_emb_conv = ConvEmb(in_channels=config.glove_char_dim, out_channels=config.glove_char_dim, kernel_size=5)

        glove_dim = config.glove_char_dim+config.glove_word_dim

        self.context_highway = CnnHighway(num_layers=config.hw_layers,
                                          input_shape=(glove_dim, config.context_limit),
                                          out_channels=glove_dim,
                                          kernel_size=config.hw_kernel,
                                          stride=config.hw_stride,
                                          dropout=config.highway_dropout)

        self.question_highway = CnnHighway(num_layers=config.hw_layers,
                                           input_shape=(glove_dim, config.question_limit),
                                           out_channels=glove_dim,
                                           kernel_size=config.hw_kernel,
                                           stride=config.hw_stride,
                                           dropout=config.highway_dropout)
        print('created context highways')
        # highway does not change the shape of the data
        # input shape should still be (batch_size, hidden_size, context/question_limit, glove_char_dim+glove_word_dim)
        self.map_highway_to_d_model = nn.Conv1d(in_channels=glove_dim, out_channels=config.d_model, kernel_size=1)
        
        context_input_shape = (config.d_model, config.context_limit)
        question_input_shape = (config.d_model, config.question_limit)

        self.stacked_embedding_encoder_block1 = StackedEncoderBlock(num_encoder_blocks=config.num_emb_blocks,
                                                                    num_conv_blocks=config.num_emb_conv,
                                                                    kernel_size=config.emb_kernel,
                                                                    out_channels = config.d_model,
                                                                    input_shape=context_input_shape,
                                                                    depthwise=config.emb_depthwise,
                                                                    num_heads = config.mod_num_heads,
                                                                    norm='batch',
                                                                    layer_dropout=config.layer_dropout,
                                                                    general_dropout=config.general_dropout)


        print("created stacked emb encoder blocks")
        # context_input_shape = (config.hidden_size, config.context_limit)
        # question_input_shape = (config.hidden_size, config.question_limit)
        # context and question input shapes should not be changing after stacked embedding encoder block


        self.context_query_attention = ContextQueryAttention(config=config, C_shape=context_input_shape, Q_shape=question_input_shape, dropout=config.cqa_dropout)

        attention_shape = (config.d_model*4, config.context_limit)

        self.map_attention_to_d_model = nn.Conv1d(in_channels=attention_shape[0], out_channels=config.d_model, kernel_size=1)

        input_shape = (config.d_model, config.context_limit)

        self.stacked_model_encoder_blocks1 = StackedEncoderBlock(num_encoder_blocks=config.num_mod_blocks,
                                                                num_conv_blocks=config.num_mod_conv,
                                                                kernel_size=config.mod_kernel,
                                                                out_channels=config.d_model,
                                                                input_shape=input_shape,
                                                                depthwise=config.mod_depthwise,
                                                                num_heads=config.mod_num_heads,
                                                                norm = 'batch',
                                                                layer_dropout = config.layer_dropout,
                                                                general_dropout=config.general_dropout)


        self.start_prob_linear = LinearLogit(in_features=config.d_model*2, out_features=1)
        self.end_prob_linear = LinearLogit(in_features=config.d_model*2, out_features=1)


    def forward(self, context_word_idxs, context_char_idxs, question_word_idxs, question_char_idxs):

        context_word_emb = self.word_embedding(context_word_idxs)
        context_char_emb = self.char_embedding(context_char_idxs)
        question_word_emb = self.word_embedding(question_word_idxs)
        question_char_emb = self.char_embedding(question_char_idxs)

        context_word_emb = self.context_word_embed_dropout(context_word_emb)
        context_char_emb = self.context_char_embed_dropout(context_char_emb)
        question_word_emb = self.question_word_embed_dropout(question_word_emb)
        question_char_emb = self.question_char_embed_dropout(question_char_emb)

        #reshapd char embeddings
        batchsize = context_word_emb.shape[0]

        context_char_emb = self.char_emb_conv(context_char_emb)  #  [B*CL, d_char, ?]
        question_char_emb = self.char_emb_conv(question_char_emb)

        context_char_emb, c_idx  = torch.max(context_char_emb, dim=2)  #new shape = [B*CL, glove_char_dim]
        question_char_emb, q_idx = torch.max(question_char_emb, dim=2)

        context_word_mask = torch.gt(torch.abs(torch.sum(context_word_emb, dim=2)), 0)
        question_word_mask = torch.gt(torch.abs(torch.sum(question_word_emb, dim=2)), 0)
        # During evaluation, variables must be volatile otherwise memory will run out
        context_embedding = torch.cat((context_word_emb, context_char_emb.view(batchsize, self.context_limit, self.glove_char_dim)), dim=2).permute(0, 2, 1)  # for conv input
        question_embedding = torch.cat((question_word_emb, question_char_emb.view(batchsize, self.question_limit, self.glove_char_dim)), dim=2).permute(0, 2, 1)

        context_highway = self.context_highway(context_embedding)
        question_highway = self.question_highway(question_embedding)

        context_path = self.map_highway_to_d_model(context_highway)
        question_path = self.map_highway_to_d_model(question_highway)

        if self.encoder_masking_flag:
            context_path = self.stacked_embedding_encoder_block1(x=context_path, mask=context_word_mask)
            question_path = self.stacked_embedding_encoder_block1(x=question_path, mask=question_word_mask)
        else:
            context_path = self.stacked_embedding_encoder_block1(x=context_path)
            question_path = self.stacked_embedding_encoder_block1(x=question_path)

        if self.c2q_masking_flag:
            attention = self.context_query_attention(context=context_path,
                                                     question=question_path,
                                                     context_word_mask=context_word_mask,
                                                     question_word_mask=question_word_mask)

        else:
            attention = self.context_query_attention(context=context_path, question=question_path)

        attention = torch.cat(attention, dim=1)
 
        atten_transform = self.map_attention_to_d_model(attention)
        if self.encoder_masking_flag:
            M0 = self.stacked_model_encoder_blocks1(x=atten_transform, mask=context_word_mask)
            M1 = self.stacked_model_encoder_blocks1(x=M0, mask=context_word_mask)
            M2 = self.stacked_model_encoder_blocks1(x=M1, mask=context_word_mask)
        else:
            M0 = self.stacked_model_encoder_blocks1(x=atten_transform)
            M1 = self.stacked_model_encoder_blocks1(x=M0)
            M2 = self.stacked_model_encoder_blocks1(x=M1)


        start_path = torch.cat([M0, M1], dim=1)
        end_path = torch.cat([M0, M2], dim=1)

        start_pred = self.start_prob_linear(start_path)
        end_pred = self.end_prob_linear(end_path)
        #
        # start_prob = F.softmax(self.start_prob_fc(start_path), dim=1)
        # end_prob = F.softmax(self.end_prob_fc(end_path), dim=1)

        return start_pred, end_pred, context_word_mask

