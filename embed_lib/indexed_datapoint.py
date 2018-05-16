import numpy as np

class IndexedDatapoint:
    def __init__(self, config_input):
        self.config = config_input

        self.context_word_idxs = np.zeros((self.config.context_limit),dtype=np.int32)
        self.context_char_idxs = np.zeros((self.config.context_limit, self.config.char_limit), dtype=np.int32)
        self.question_word_idxs = np.zeros((self.config.question_limit), dtype=np.int32)
        self.question_char_idxs = np.zeros((self.config.question_limit, self.config.char_limit), dtype=np.int32)
        self.context_true_answer_start = np.zeros(1, dtype=np.int32)
        self.context_true_answer_end = np.zeros(1, dtype=np.int32)
        self.id = 0


