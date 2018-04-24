from encoder_block_lib.stacked_encoder_blocks import StackedEncoderBlock
from context_query_attention_lib.context_query_attention import ContextQueryAttention
from highway_lib.cnn_highway import CnnHighway
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



class QANet(nn.Module):

    def __init__(self, config):
        super(QANet, self).__init__()

        context_input_shape = (config.glove_char_dim+config.glove_word_dim, config.context_limit)
        question_input_shape = (config.glove_char_dim+config.glove_word_dim, config.question_limit)

        self.context_highway = CnnHighway(num_layers=config.hw_layers,
                                          input_size=context_input_shape,
                                          hidden_size=context_input_shape[0],
                                          kernel_size=config.hw_kernel,
                                          stride=config.hw_stride)

        self.question_highway = CnnHighway(num_layers=config.hw_layers,
                                           input_size=question_input_shape,
                                           hidden_size=context_input_shape[0],
                                           kernel_size=config.hw_kernel,
                                           stride=config.hw_stride)
        print('created context highways')
        # highway does not change the shape of the data
        # input shape should still be (batch_size, hidden_size, context/question_limit, glove_char_dim+glove_word_dim)
        self.context_highway_to_hidden_conv = nn.Conv1d(in_channels=context_input_shape[0], out_channels=config.hidden_size, kernel_size=1)
        self.question_highway_to_hidden_conv = nn.Conv1d(in_channels=question_input_shape[0], out_channels=config.hidden_size, kernel_size=1)
        context_input_shape = (config.hidden_size, config.context_limit)
        question_input_shape = (config.hidden_size, config.question_limit)

        self.context_stacked_embedding_encoder_block1 = StackedEncoderBlock(num_encoder_blocks=config.num_emb_blocks,
                                                                            num_conv_layers=config.num_emb_conv,
                                                                            kernel_size=config.emb_kernel,
                                                                            hidden_size = config.hidden_size,
                                                                            input_shape=context_input_shape,
                                                                            depthwise=config.emb_depthwise)

        self.question_stacked_embedding_encoder_block1 = StackedEncoderBlock(num_encoder_blocks=config.num_emb_blocks,
                                                                             num_conv_layers=config.num_emb_conv,
                                                                             kernel_size=config.emb_kernel,
                                                                             hidden_size=config.hidden_size,
                                                                             input_shape=question_input_shape,
                                                                             depthwise=config.emb_depthwise)

        print("created stacked emb encoder blocks")
        # context_input_shape = (config.hidden_size, config.context_limit)
        # question_input_shape = (config.hidden_size, config.question_limit)
        # context and question input shapes should not be changing after stacked embedding encoder block


        self.context_query_attention = ContextQueryAttention(config=config, C_shape=context_input_shape, Q_shape=question_input_shape)
        print('created query attention')
        attention_shape = (config.hidden_size*4, config.context_limit)

        self.attention_to_hidden_size_conv = nn.Conv1d(in_channels=attention_shape[0], out_channels=config.hidden_size, kernel_size=1)

        input_shape = (config.hidden_size, config.context_limit)

        self.stacked_model_encoder_blocks1 = StackedEncoderBlock(num_encoder_blocks=config.num_mod_blocks,
                                                                num_conv_layers=config.num_mod_conv,
                                                                kernel_size=config.mod_kernel,
                                                                hidden_size=config.hidden_size,
                                                                input_shape=input_shape,
                                                                depthwise=config.mod_depthwise)


        # self.stacked_model_encoder_blocks2 = StackedEncoderBlock(num_encoder_blocks=config.num_mod_blocks,
        #                                                         num_conv_layers=config.num_mod_conv,
        #                                                         kernel_size=config.mod_kernel,
        #                                                         hidden_size=config.hidden_size,
        #                                                         input_shape=input_shape)
        #
        # self.stacked_model_encoder_blocks3 = StackedEncoderBlock(num_encoder_blocks=config.num_mod_blocks,
        #                                                         num_conv_layers=config.num_mod_conv,
        #                                                         kernel_size=config.mod_kernel,
        #                                                         hidden_size=config.hidden_size,
        #                                                         input_shape=input_shape)

        input_shape = (config.hidden_size, config.context_limit)
        total_features = input_shape[0] * input_shape[1] *2
        self.start_prob_fc = nn.Linear(in_features=total_features, out_features=config.context_limit)
        self.end_prob_fc = nn.Linear(in_features=total_features, out_features=config.context_limit)


    def forward(self, context_embedding, question_embedding):

        context_highway = self.context_highway(context_embedding)
        question_highway = self.question_highway(question_embedding)

        context_highway = self.context_highway_to_hidden_conv(context_highway)
        question_highway = self.question_highway_to_hidden_conv(question_highway)

        context_path = self.context_stacked_embedding_encoder_block1(context_highway)
        question_path = self.question_stacked_embedding_encoder_block1(question_highway)

        attention = self.context_query_attention(context_path, question_path)
        #print('attention', attention)
        attention = torch.cat(attention, dim=1)
        #print('new attention shape', attention.shape)
        #
        atten_transform = self.attention_to_hidden_size_conv(attention)

        enc1 = self.stacked_model_encoder_blocks1(atten_transform)
        enc2 = self.stacked_model_encoder_blocks1(enc1)
        enc3 = self.stacked_model_encoder_blocks1(enc2)


        # enc1 = self.stacked_model_encoder_blocks1(attention)
        # enc2 = self.stacked_model_encoder_blocks2(enc1)
        # enc3 = self.stacked_model_encoder_blocks3(enc2)
        # #
        # #
        batch_size = enc1.shape[0]
        total_features = enc1.shape[1]*enc1.shape[2]*2
        start_path = torch.cat([enc1, enc2], dim=1).view(batch_size, total_features)
        end_path = torch.cat([enc1, enc3], dim=1).view(batch_size, total_features)

        start_pred = self.start_prob_fc(start_path)
        end_pred = self.end_prob_fc(end_path)
        #
        # start_prob = F.softmax(self.start_prob_fc(start_path), dim=1)
        # end_prob = F.softmax(self.end_prob_fc(end_path), dim=1)

        return start_pred, end_pred

