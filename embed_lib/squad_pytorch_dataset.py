import torch
import numpy as np
from torch.utils.data.dataset import Dataset
class SquadPytorchDataset(Dataset):
    def __init__(self, squad_emb):
        self.datapoints = squad_emb.datapoints

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):

        qa_id = self.datapoints.id[idx]

        context_word_idxs = self.datapoints.context_word_idxs[idx]
        context_char_idxs = self.datapoints.context_char_idxs[idx]

        question_word_idxs = self.datapoints.question_word_idxs[idx]
        question_char_idxs = self.datapoints.question_char_idxs[idx]

        context_true_answer_start = self.datapoints.context_true_answer_start[idx]
        context_true_answer_end = self.datapoints.context_true_answer_end[idx]
        
        return {'context_word_idxs': context_word_idxs,
                'context_char_idxs': context_char_idxs,
                'question_word_idxs': question_word_idxs,
                'question_char_idxs': question_char_idxs,
                'context_true_answer_start': context_true_answer_start,
                'context_true_answer_end': context_true_answer_end,
                'qa_id': qa_id}
