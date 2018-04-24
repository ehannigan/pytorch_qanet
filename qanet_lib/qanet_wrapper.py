from qanet_lib.qanet import QANet
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
class QANetWrapper:
    def __init__(self, config):
        print('in QANetWrapper')
        self.config = config
        self.net = QANet(config)
        if self.config.cuda_flag:
            self.net = self.net.cuda()
        self.__get_optimizer()
        self.__get_criterion()

    def __get_optimizer(self):
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config.learning_rate)

    def __get_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def train(self, trainloader):
        print('starting train')
        self.net.train()
        num_channels = 1
        total_glove_dim = self.config.glove_word_dim + self.config.glove_char_dim

        for epoch in range(self.config.num_epochs):
            running_loss = 0
            running_f1 = 0
            for i, batch in enumerate(trainloader):

                context_word_emb = batch['context_word_emb']  #shape = [N, CL, Dim]
                context_char_emb = batch['context_char_emb']
                question_word_emb = batch['question_word_emb']
                question_char_emb = batch['question_char_emb']
                answer_start_idx = batch['answer_start_idx']
                #print('answer start idx.shape', answer_start_idx.shape)
                answer_end_idx = batch['answer_end_idx']
                qa_id = batch['qa_id']


                context_embedding = Variable(torch.cat((context_word_emb, context_char_emb), dim=2)).permute(0,2,1) #for conv input
                question_embedding = Variable(torch.cat((question_word_emb, question_char_emb), dim=2)).permute(0,2,1)
                answer_start_idx = Variable(answer_start_idx.long())
                answer_end_idx = Variable(answer_end_idx.long())
                if self.config.cuda_flag:
                    context_embedding = context_embedding.cuda()
                    question_embedding = question_embedding.cuda()
                    answer_start_idx = answer_start_idx.cuda()
                    answer_end_idx = answer_end_idx.cuda()


                # context_embedding = context_embedding.view((self.config.batch_size, num_channels, self.config.context_limit, total_glove_dim)).contiguous()
                # question_embedding = question_embedding.view((self.config.batch_size, num_channels, self.config.question_limit, total_glove_dim)).contiguous()

                assert context_embedding.shape[1:] == (total_glove_dim, self.config.context_limit)
                assert question_embedding.shape[1:] == (total_glove_dim, self.config.question_limit)


                self.optimizer.zero_grad()
                start_pred, end_pred = self.net.forward(context_embedding, question_embedding)

                #print('start pred.shape', start_pred)
                #print('answer_start_idx.shape', answer_start_idx)
                loss = self.__get_loss(start_pred, answer_start_idx, end_pred, answer_end_idx)

                loss.backward()
                running_loss += loss
                if (i%self.config.print_freq) == (self.config.print_freq-1):
                    cur_loss = running_loss/self.config.print_freq
                    cur_f1 = running_f1/self.config.print_freq
                    print('epoch: {} i: {} || loss: {}'.format(epoch+1, self.config.print_freq, cur_loss))
                    running_loss = 0
                    running_f1 = 0
                self.optimizer.step()

        self.save_checkpoint()


    def __get_loss(self, start_pred, answer_start_idx, end_pred, answer_end_idx):
        loss1 = self.criterion(start_pred, answer_start_idx)
        #print('loss1.shape', loss1.shape)

        loss2 = self.criterion(end_pred, answer_end_idx)
        loss = torch.cat([loss1, loss2], dim=0)
        loss = torch.mean(loss, dim=0)
        return loss

    def get_answer_text(self, datapoint_dict, qa_id, start_pred_idx, end_pred_idx):
        answer_dict = {}
        #remapped_dict = {}
        for i, id in enumerate(qa_id):
            raw_datapoint = datapoint_dict[str(id)]
            context = raw_datapoint.context
            span = raw_datapoint.context_spans


            start_word_idx = int(start_pred_idx[i].data[0])
            end_word_idx = int(end_pred_idx[i].data[0])
            print('i: {}, qid: {}, start_word_idx: {}, end_word_idx: {}, len(span): {}'.format(i, id, start_word_idx, end_word_idx, len(span)))

            start_char_idx = span[start_word_idx][0]
            end_char_idx = span[end_word_idx][1]
            answer_dict[str(id)] = context[start_char_idx: end_char_idx]
            #remapped_dict[uuid] = context[start_idx: end_idx]
        return answer_dict #, remapped_dict


    def test(self, testloader, train_raw):
        self.net.eval()
        running_loss = 0
        count = 0
        pred_answer_dict = {}
        for i, batch in enumerate(testloader):
            context_word_emb = batch['context_word_emb']  # shape = [N, CL, Dim]
            context_char_emb = batch['context_char_emb']
            question_word_emb = batch['question_word_emb']
            question_char_emb = batch['question_char_emb']
            answer_start_idx = batch['answer_start_idx']
            #print('answer start idx.shape', answer_start_idx.shape)
            answer_end_idx = batch['answer_end_idx']
            qa_id = batch['qa_id']


            context_embedding = Variable(torch.cat((context_word_emb, context_char_emb), dim=2)).permute(0, 2,                                                                                    1)  # for conv input
            question_embedding = Variable(torch.cat((question_word_emb, question_char_emb), dim=2)).permute(0, 2, 1)
            answer_start_idx = Variable(answer_start_idx.long())
            answer_end_idx = Variable(answer_end_idx.long())
            if self.config.cuda_flag:
                context_embedding = context_embedding.cuda()
                question_embedding = question_embedding.cuda()
                answer_start_idx = answer_start_idx.cuda()
                answer_end_idx = answer_end_idx.cuda()



            self.optimizer.zero_grad()
            start_pred, end_pred = self.net.forward(context_embedding, question_embedding)
            _, start_pred_idx = torch.max(start_pred, dim=0)
            _, end_pred_idx = torch.max(end_pred, dim=0)
            print('test qaid', qa_id)
            cur_pred_answer_dict = self.get_answer_text(train_raw.datapoint_dict, qa_id, start_pred_idx, end_pred_idx)
            pred_answer_dict.update(cur_pred_answer_dict)
            # f1 = self.get_f1(start_pred, answer_start_idx, end_pred, answer_end_idx)

            loss = self.__get_loss(start_pred, answer_start_idx, end_pred, answer_end_idx)
            running_loss += loss
            count += 1
        total_loss = running_loss/count


        return total_loss, pred_answer_dict


    def save_checkpoint(self):
        state_dict = self.net.state_dict()
        optim_dict = self.optimizer.state_dict()
        checkpoint = {'state_dict': state_dict,
                      'optimizer_dict': optim_dict}
        torch.save(checkpoint, 'checkpoint.pth')


    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
