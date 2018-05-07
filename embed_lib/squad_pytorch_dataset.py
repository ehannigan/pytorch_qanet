import torch
import numpy as np
from torch.utils.data.dataset import Dataset
class SquadPytorchDataset(Dataset):
    def __init__(self, squad_emb):
        self.squad_emb = squad_emb

    def __len__(self):
        return len(self.squad_emb.datapoints)

    def __getitem__(self, idx):

        qa_id = self.squad_emb.get_id(idx)

        context_word_emb = self.squad_emb.get_context_word_emb(idx)
        context_char_emb = self.squad_emb.get_context_char_emb(idx)

        question_word_emb = self.squad_emb.get_question_word_emb(idx)
        question_char_emb = self.squad_emb.get_question_char_emb(idx)

        answer_start_idx = self.squad_emb.get_answer_start_idx(idx)
        answer_end_idx = self.squad_emb.get_answer_end_idx(idx)
        
        return {'context_word_emb': context_word_emb,
                'context_char_emb': context_char_emb,
                'question_word_emb': question_word_emb,
                'question_char_emb': question_char_emb,
                'answer_start_idx': answer_start_idx,
                'answer_end_idx': answer_end_idx,
                'qa_id': qa_id}
