from embed_lib.emb_datatpoint import EmbDatapoint
import numpy as np
class SquadEmb:
    def __init__(self, SquadRaw, word_embedding, char_embedding, config):
        self.config = config
        self.word_emb = word_embedding
        self.char_emb = char_embedding
        self.word_emb_dim = self.word_emb.emb_mat.shape[1]
        self.char_emb_dim = self.char_emb.emb_mat.shape[1]

        self.datapoints = []
        self.__create_datapoints(SquadRaw)

    def __get_word_idx(self, word):
        for w in (word, word.lower(), word.capitalize(), word.upper()):
            if w in self.word_emb.token2idx_dict:
                return self.word_emb.token2idx_dict[w]
            return 1  # OOV


    def __get_char_idx(self, char):
        if char in self.char_emb.token2idx_dict:
            return self.char_emb.token2idx_dict[char]
        return 1  # OOV


    def __check_datapoint_size(self, example):
        context_len = len(example.context_tokens)
        question_len = len(example.question_tokens)
        max_answer_len = example.answer_end_idxs[0] - example.answer_start_idxs[0]

        if (context_len > self.config.context_limit) or \
                (question_len > self.config.question_limit) or \
                (max_answer_len > self.config.answer_limit):
            return False
        return True


    def __create_datapoints(self, SquadRaw):

        for id, raw_datapoint in SquadRaw.datapoint_dict.items():

            datapoint_valid = self.__check_datapoint_size(raw_datapoint)
            if not datapoint_valid:
                break

            emb_datapoint = EmbDatapoint(self.config)


            for i, word in enumerate(raw_datapoint.context_tokens):
                emb_datapoint.context_idxs[i] = self.__get_word_idx(word)

            for i, word in enumerate(raw_datapoint.question_tokens):
                emb_datapoint.question_idxs[i] = self.__get_word_idx(word)

            for i, word in enumerate(raw_datapoint.context_chars):
                for j, char in enumerate(word):
                    if j < self.config.char_limit:
                        emb_datapoint.context_char_idxs[i, j] = self.__get_char_idx(char)
                    else:
                        break

            for i, word in enumerate(raw_datapoint.question_chars):
                for j, char in enumerate(word):
                    if j < self.config.char_limit:
                        emb_datapoint.context_char_idxs[i, j] = self.__get_char_idx(char)
                    else:
                        break

            start_idx = raw_datapoint.answer_start_idxs[-1]
            end_idx = raw_datapoint.answer_end_idxs[-1]


            emb_datapoint.context_true_answer_start = start_idx
            emb_datapoint.context_true_answer_end = end_idx
            emb_datapoint.id = raw_datapoint.uuid

            self.datapoints.append(emb_datapoint)




    def get_context_word_emb(self, idx):
        """ for question idx
        :param idx:
        :return:
        """
        context_idxs = self.datapoints[idx].context_idxs
        num_words = self.config.context_limit
        context_word_emb = np.zeros((num_words, self.word_emb_dim), dtype=np.float32)
        for i, word_idx in enumerate(context_idxs):
            context_word_emb[i, :] = self.word_emb.emb_mat[word_idx, :]
        return context_word_emb

    def get_context_char_emb(self, idx):
        context_char_idxs = self.datapoints[idx].context_char_idxs
        num_words = self.config.context_limit
        num_chars = self.config.char_limit
        context_char_emb = np.zeros((num_words, num_chars, self.char_emb_dim), dtype=np.float32)
        for i, word_idx in enumerate(context_char_idxs):
            for j, char_idx in enumerate(word_idx):
                context_char_emb[i, j, :] = self.char_emb.emb_mat[char_idx, :]
        context_char_emb = np.amax(context_char_emb, axis=1)
        return context_char_emb

    def get_question_word_emb(self, idx):
        question_idxs = self.datapoints[idx].question_idxs
        num_words = self.config.question_limit
        question_word_emb = np.zeros((num_words, self.word_emb_dim), dtype=np.float32)
        for i, word_idx in enumerate(question_idxs):
            question_word_emb[i, :] = self.word_emb.emb_mat[word_idx, :]
        return question_word_emb

    def get_question_char_emb(self, idx):
        question_char_idxs = self.datapoints[idx].question_char_idxs
        num_words = self.config.question_limit
        num_chars = self.config.char_limit
        question_char_emb = np.zeros((num_words, num_chars, self.char_emb_dim), dtype=np.float32)
        for i, word_idx in enumerate(question_char_idxs):
            for j, char_idx in enumerate(word_idx):
                question_char_emb[i, j, :] = self.char_emb.emb_mat[char_idx, :]
        question_char_emb = np.amax(question_char_emb, axis=1)
        return question_char_emb

    def get_answer_start_idx(self, idx):
        answer_start_idx = self.datapoints[idx].context_true_answer_start
        return answer_start_idx

    def get_answer_end_idx(self, idx):
        answer_end_idx = self.datapoints[idx].context_true_answer_end
        return answer_end_idx

    def get_id(self, idx):
        id = self.datapoints[idx].id
        return id



