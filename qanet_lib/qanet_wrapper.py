from qanet_lib.qanet import QANet
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from train_val_plotter import Plotter
import torch.nn.functional as F
from helper_functions import exponential_mask
import math
import os
from helper_functions import evaluate_predictions
from helper_functions import metric_max_over_ground_truths
from helper_functions import exact_match_score, f1_score
import gc

class QANetWrapper:
    def __init__(self, config, embedding, checkpoint_path):
        print('in QANetWrapper')
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.net = QANet(config=config, word_emb_weights=torch.FloatTensor(embedding['word_emb_weights']), char_emb_weights=torch.FloatTensor(embedding['char_emb_weights']))
        if self.config.cuda_flag:
            self.net = self.net.cuda()
        self.optimizer = self.__get_optimizer(config.optimizer_name)
        self.criterion = self.__get_criterion(config.criterion_name)
        self.loss_plotter = Plotter(config, train_name='train_loss', val_name='val_loss')
        self.f1_plotter = Plotter(config, train_name='train_f1', val_name='val_f1')
        self.EM_plotter = Plotter(config, train_name='train_em', val_name='val_em')
        self.cur_epoch = 0
        self.global_step = 0

    def __get_optimizer(self, name):
        if name == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=self.config.learning_rate)
        elif name == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=self.config.learning_rate, betas=(.8, .999), eps=1e-7)
        else:
            raise ValueError("optimizer name is not recognized")
        return optimizer

    def __get_criterion(self, name):
        if name == 'cross_entropy_loss':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("criterion name is not recognized")
        return criterion


    def __learning_rate_warm_up(self, optimizer, global_step):
        max_learning_rate = self.config.learning_rate
        num_warm_up_steps = self.config.num_learning_rate_warm_up_steps

        new_learning_rate = min(max_learning_rate, max_learning_rate/math.log(num_warm_up_steps)*math.log(global_step + 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate
        return optimizer

    def train(self, trainloader, train_raw=None, valloader=None, val_raw=None):
        print('starting train')
        self.net.train()
        volatile_flag=False

        for epoch in range(self.cur_epoch, self.cur_epoch+self.config.num_epochs):
            running_loss = 0

            for i, batch in enumerate(trainloader):
                self.net.train()
                if self.global_step < self.config.num_learning_rate_warm_up_steps:
                    self.optimizer = self.__learning_rate_warm_up(optimizer=self.optimizer, global_step=self.global_step)
                self.optimizer.zero_grad()

                context_word_idxs, context_char_idxs, question_word_idxs, question_char_idxs, answer_start_idx, answer_end_idx, qa_id = self.__extract_from_batch(batch=batch, volatile_flag=volatile_flag)
                start_logit, end_logit, context_word_mask = self.net.forward(context_word_idxs=context_word_idxs,
                                                                           context_char_idxs=context_char_idxs,
                                                                           question_word_idxs=question_word_idxs,
                                                                           question_char_idxs=question_char_idxs)


                if self.config.pred_mask:
                    start_logit_masked, end_logit_masked = self.__mask_output_logits(start_logit=start_logit,
                                                                                     end_logit=end_logit,
                                                                                     mask=context_word_mask)
                    p1 = F.softmax(start_logit_masked, dim=1)
                    p2 = F.softmax(end_logit_masked, dim=1)
                    loss = self.__get_loss(start_logit=start_logit_masked,
                                           end_logit=end_logit_masked,
                                           answer_start_idx=answer_start_idx,
                                           answer_end_idx=answer_end_idx)
                else:
                    p1 = F.softmax(start_logit, dim=1)
                    p2 = F.softmax(end_logit, dim=1)
                    loss = self.__get_loss(start_logit=start_logit,
                                           end_logit=end_logit,
                                           answer_start_idx=answer_start_idx,
                                           answer_end_idx=answer_end_idx)
                #print('loss.shape', loss.shape)
                loss.backward()

                # for f in self.net.parameters():
                #     print('data is')
                #     print(f.data)
                #     print('grad is')
                #     print(f.grad)

                self.optimizer.step()
                self.global_step += 1


                running_loss += loss.data[0]
                if (i%self.config.print_freq) == (self.config.print_freq-1):
                    cur_loss = running_loss/self.config.print_freq
                    predictions = self.get_prediction_dict(train_raw.datapoint_dict, qa_id, p1, p2)
                    train_eval = self.evaluate_on_batch(train_raw.datapoint_dict, predictions)
                    self.loss_plotter.update_train_lists(self.global_step, cur_loss)
                    self.f1_plotter.update_train_lists(self.global_step, train_eval['f1'])
                    self.EM_plotter.update_train_lists(self.global_step, train_eval['exact_match'])
                    if valloader:
                        val_loss, predictions = self.evaluate(valloader, val_raw)
                        val_eval = self.evaluate_on_batch(val_raw.datapoint_dict, predictions)
                        self.loss_plotter.update_val_lists(val_loss)
                        self.f1_plotter.update_val_lists(val_eval['f1'])
                        self.EM_plotter.update_val_lists(val_eval['exact_match'])
                        print('epoch: {} gs: {} || train_loss: {}, val_loss: {}, train_f1: {}, val_f1: {}'.format(epoch+1, self.global_step+1, cur_loss, val_loss, train_eval['f1'], val_eval['f1']))
                    else:
                        print('epoch: {} gs: {} || train_loss: {}, train_f1: {}'.format(epoch + 1, self.global_step + 1, cur_loss, train_eval['f1']))
                    running_loss = 0

            self.save_checkpoint(epoch, self.global_step)

    def __get_loss(self, start_logit, end_logit, answer_start_idx, answer_end_idx):

        loss1 = self.criterion(start_logit, answer_start_idx)
        loss2 = self.criterion(end_logit, answer_end_idx)
        loss = loss1+loss2
        #loss = torch.mean(loss, dim=0)
        return loss

    def get_inference_indices(self, start_pred, end_pred, context_len):
        inference_mat = torch.mm(start_pred.unsqueeze(1), end_pred.unsqueeze(0))
        upper_inference = torch.triu(inference_mat)
        inference_answer_limit = torch.tril(upper_inference, self.config.answer_limit)
        best_val_by_row, best_col = torch.max(inference_answer_limit, dim=0)
        best_val, p2 = torch.max(best_val_by_row, dim=0)
        best_val_by_col, best_row = torch.max(inference_answer_limit, dim=1)
        bet_val, p1 = torch.max(best_val_by_col, dim=0)

        # inference_mat = torch.mm(start_pred.unsqueeze(1), end_pred.unsqueeze(0)).data.cpu().numpy()
        # upper_inference = np.triu(inference_mat)
        # inference_answer_limit = np.tril(upper_inference, self.config.answer_limit)
        # p1_np, p2_np = np.unravel_index(inference_answer_limit.argmax(), inference_answer_limit.shape)
        # p1_np = p1, p2_np = p2

        return (p1, p2)

    def get_prediction_dict(self, datapoint_dict, qa_id, p1, p2):
        prediction_dict = {}
        #remapped_dict = {}
        for q_id_i, p1_i, p2_i in zip(qa_id, p1, p2):
            raw_datapoint = datapoint_dict[str(q_id_i)]
            context = raw_datapoint.context
            span = raw_datapoint.context_spans
            start_pred_idx, end_pred_idx = self.get_inference_indices(p1_i, p2_i, len(span))
            start_word_idx = start_pred_idx.data[0]
            end_word_idx = end_pred_idx.data[0]
            #print('qid: {}, start_word_idx: {}, end_word_idx: {}, len(span): {}'.format(q_id_i, start_word_idx, end_word_idx, len(span)))
            start_char_idx = span[start_word_idx][0]
            end_char_idx = span[end_word_idx][1]
            prediction_dict[str(q_id_i)] = context[start_char_idx:end_char_idx]

            #remapped_dict[uuid] = context[start_idx: end_idx]
        return prediction_dict #, remapped_dict

    def evaluate_on_batch(self, datapoint_dict, prediction_dict):
        exact_match = 0
        f1 = 0
        total = 0
        for qa_id, prediction in prediction_dict.items():
            total += 1
            ground_truths = datapoint_dict[qa_id].answer_texts
            exact_match += metric_max_over_ground_truths(metric_fn=exact_match_score, prediction=prediction, ground_truths=ground_truths)
            f1 += metric_max_over_ground_truths(metric_fn=f1_score, prediction=prediction, ground_truths=ground_truths)
        f1 = f1/total * 100
        exact_match = exact_match/total * 100
        return {'f1': f1, 'exact_match': exact_match}

    def evaluate(self, evaloader, eval_raw):
        self.net.eval()
        volatile_flag = True
        running_loss = 0
        count = 0
        prediction_dict = {}
        for i, batch in enumerate(evaloader):

            context_word_idxs, context_char_idxs, question_word_idxs, question_char_idxs, answer_start_idx, answer_end_idx, qa_id = self.__extract_from_batch(batch=batch, volatile_flag=volatile_flag)

            start_logit, end_logit, context_word_mask = self.net.forward(context_word_idxs=context_word_idxs,
                                                                        context_char_idxs=context_char_idxs,
                                                                        question_word_idxs=question_word_idxs,
                                                                        question_char_idxs=question_char_idxs)
            if self.config.pred_mask:
                start_logit_masked, end_logit_masked = self.__mask_output_logits(start_logit=start_logit, end_logit=end_logit, mask=context_word_mask)
                p1 = F.softmax(start_logit_masked, dim=1)
                p2 = F.softmax(end_logit_masked, dim=1)
                loss = self.__get_loss(start_logit=start_logit_masked,
                                   end_logit=end_logit_masked,
                                   answer_start_idx=answer_start_idx,
                                   answer_end_idx=answer_end_idx)
            else:
                p1 = F.softmax(start_logit, dim=1)
                p2 = F.softmax(end_logit, dim=1)
                loss = self.__get_loss(start_logit=start_logit,
                                        end_logit=end_logit,
                                        answer_start_idx=answer_start_idx,
                                        answer_end_idx=answer_end_idx)


            cur_prediction_dict = self.get_prediction_dict(eval_raw.datapoint_dict, qa_id, p1, p2)
            prediction_dict.update(cur_prediction_dict)

            running_loss += loss.data[0]
            count += 1
            if count >= self.config.max_val_batches:
                break
        total_loss = running_loss / count

        return total_loss, prediction_dict

    def __extract_from_batch(self, batch, volatile_flag=False):


        context_word_idxs = batch['context_word_idxs']  # shape = [N, CL, Dim]
        context_char_idxs = batch['context_char_idxs']
        question_word_idxs = batch['question_word_idxs']
        question_char_idxs = batch['question_char_idxs']
        answer_start_idx = batch['context_true_answer_start']
        # print('answer start idx.shape', answer_start_idx.shape)
        answer_end_idx = batch['context_true_answer_end']
        qa_id = batch['qa_id']

        context_word_idxs = Variable(context_word_idxs, volatile=volatile_flag)
        context_char_idxs = Variable(context_char_idxs, volatile=volatile_flag)
        question_word_idxs = Variable(question_word_idxs, volatile=volatile_flag)
        question_char_idxs = Variable(question_char_idxs, volatile=volatile_flag)
        answer_start_idx = Variable(answer_start_idx.long(), volatile=volatile_flag)
        answer_end_idx = Variable(answer_end_idx.long(), volatile=volatile_flag)
        if self.config.cuda_flag:
            context_word_idxs = context_word_idxs.cuda()
            context_char_idxs = context_char_idxs.cuda()
            question_word_idxs = question_word_idxs.cuda()
            question_char_idxs = question_char_idxs.cuda()
            answer_start_idx = answer_start_idx.cuda()
            answer_end_idx = answer_end_idx.cuda()

        return context_word_idxs, context_char_idxs, question_word_idxs, question_char_idxs, answer_start_idx, answer_end_idx, qa_id


    def predict(self, testloader, test_raw):
        self.net.eval()
        volatile_flag = True
        running_loss = 0
        count = 0
        prediction_dict = {}
        for i, batch in enumerate(testloader):

            context_word_idxs, context_char_idxs, question_word_idxs, question_char_idxs, answer_start_idx, answer_end_idx, qa_id = self.__extract_from_batch(batch=batch, volatile_flag=volatile_flag)

            start_logit, end_logit, context_word_mask = self.net.forward(context_word_idxs=context_word_idxs,
                                                                        context_char_idxs=context_char_idxs,
                                                                        question_word_idxs=question_word_idxs,
                                                                        question_char_idxs=question_char_idxs)


            if self.config.pred_mask:
                start_logit_masked, end_logit_masked = self.__mask_output_logits(start_logit=start_logit, end_logit=end_logit, mask=context_word_mask)
                p1 = F.softmax(start_logit_masked, dim=1)
                p2 = F.softmax(end_logit_masked, dim=1)
                loss = self.__get_loss(start_logit=start_logit_masked,
                                   end_logit=end_logit_masked,
                                   answer_start_idx=answer_start_idx,
                                   answer_end_idx=answer_end_idx)
            else:
                p1 = F.softmax(start_logit, dim=1)
                p2 = F.softmax(end_logit, dim=1)
                loss = self.__get_loss(start_logit=start_logit,
                                        end_logit=end_logit,
                                        answer_start_idx=answer_start_idx,
                                        answer_end_idx=answer_end_idx)


            cur_prediction_dict = self.get_prediction_dict(test_raw.datapoint_dict, qa_id, p1, p2)
            prediction_dict.update(cur_prediction_dict)
            # f1 = self.get_f1(start_pred, answer_start_idx, end_pred, answer_end_idx)

            #loss = self.__get_loss(start_pred_masked, answer_start_idx, end_pred_masked, answer_end_idx)
            running_loss += loss.data[0]
            count += 1

        total_loss = running_loss/count

        return total_loss, prediction_dict

    def save_checkpoint(self, epoch, global_step):
        state_dict = self.net.state_dict()
        optim_dict = self.optimizer.state_dict()
        checkpoint = {'epoch': epoch,
                      'global_step': global_step,
                      'state_dict': state_dict,
                      'optimizer_dict': optim_dict,
                      'loss_plotter': self.loss_plotter,
                      'f1_plotter': self.f1_plotter}

        torch.save(checkpoint, self.checkpoint_path.format(epoch))

    def load_model(self, checkpoint_path, epoch_no):
        checkpoint = torch.load(checkpoint_path.format(epoch_no))
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
        self.loss_plotter = checkpoint['loss_plotter']
        self.f1_plotter = checkpoint['f1_plotter']
        self.cur_epoch = checkpoint['epoch']+1
        self.global_step = checkpoint['global_step']+1

    def __mask_output_logits(self, start_logit, end_logit, mask):
        start_logit_masked = exponential_mask(tensor=start_logit, mask=mask)
        end_logit_masked = exponential_mask(tensor=end_logit, mask=mask)
        return start_logit_masked, end_logit_masked