from embed_lib.emb_datatpoint import EmbDatapoint
import numpy as np
class SquadEmb:
    def __init__(self, config, SquadRaw, word_embedder, char_embedder):
        self.config = config
        self.datapoints = []
        self.word_embedder = word_embedder
        self.char_embedder = char_embedder

        self.context_len_sum = 0
        self.question_len_sum = 0
        self.max_answer_len_sum = 0
        self.total_invalid_points = 0
        self.__create_datapoints(SquadRaw)

    def __check_datapoint_size(self, example):
        context_len = len(example.context_tokens)
        self.context_len_sum += context_len
        question_len = len(example.question_tokens)
        self.question_len_sum += question_len
        max_answer_len = example.answer_end_idxs[0] - example.answer_start_idxs[0]
        self.max_answer_len_sum += max_answer_len

        if (context_len > self.config.context_limit):
            print('context too long')
            self.total_invalid_points+=1
            return False
        if (question_len > self.config.question_limit):
            print('question too long')
            self.total_invalid_points += 1
            return False
        if (max_answer_len > self.config.answer_limit):
            print('max answer len too long')
            self.total_invalid_points += 1
            return False
        return True

    def __create_datapoints(self, SquadRaw):
        total_invalid_points=0
        for id, raw_datapoint in SquadRaw.datapoint_dict.items():

            datapoint_valid = self.__check_datapoint_size(raw_datapoint)
            if not datapoint_valid:
                continue

            emb_datapoint = EmbDatapoint(self.config)

            for i, word in enumerate(raw_datapoint.context_tokens):
                for w in (word, word.lower(), word.capitalize(), word.upper()):
                    #test all possible casings of the word w. If you find an embedding, break and assign it to context_word_idxs
                    #if not, keep trying until you run out of possible word casings. If you do not find any index for the word, assign context_word_idxs to be 1 (out of vocabulary)
                    emb = self.word_embedder.get_indexing(w)
                    if emb != 1:
                        break
                emb_datapoint.context_word_idxs[i] = emb

            for i, word in enumerate(raw_datapoint.question_tokens):
                for w in (word, word.lower(), word.capitalize(), word.upper()):
                    #test all possible casings of the word w. If you find an embedding, break and assign it to context_word_idxs
                    #if not, keep trying until you run out of possible word casings. If you do not find any index for the word, assign context_word_idxs to be 1 (out of vocabulary)
                    emb = self.word_embedder.get_indexing(w)
                    if emb != 1:
                        break
                emb_datapoint.question_word_idxs[i] = emb

            for i, word in enumerate(raw_datapoint.context_chars):
                for j, char in enumerate(word):
                    if j < self.config.char_limit:
                        emb_datapoint.context_char_idxs[i, j] = self.char_embedder.get_indexing(char)


            for i, word in enumerate(raw_datapoint.question_chars):
                for j, char in enumerate(word):
                    if j < self.config.char_limit:
                        emb_datapoint.question_char_idxs[i, j] = self.char_embedder.get_indexing(char)


            start_idx = raw_datapoint.answer_start_idxs[-1]
            end_idx = raw_datapoint.answer_end_idxs[-1]

            emb_datapoint.context_true_answer_start = start_idx
            emb_datapoint.context_true_answer_end = end_idx
            emb_datapoint.id = raw_datapoint.uuid
            self.datapoints.append(emb_datapoint)

        total_datapoints = len(SquadRaw.datapoint_dict)
        self.average_context_len = self.context_len_sum/total_datapoints
        self.average_question_len = self.question_len_sum/total_datapoints
        self.average_max_answer_len = self.max_answer_len_sum/total_datapoints

    def get_context_word_emb(self, idx):
        context_word_idxs = self.datapoints[idx].context_word_idxs
        num_words = self.config.context_limit
        context_word_emb = np.zeros((num_words, self.word_embedder.emb_dim), dtype=np.float32)
        for i, word_idx in enumerate(context_word_idxs):
            context_word_emb[i, :] = self.word_embedder.get_embedding(word_idx)
        return context_word_emb

    def get_context_char_emb(self, idx):
        context_char_idxs = self.datapoints[idx].context_char_idxs
        num_words = self.config.context_limit
        num_chars = self.config.char_limit
        context_char_emb = np.zeros((num_words, num_chars, self.char_embedder.emb_dim), dtype=np.float32)
        for i, word_idx in enumerate(context_char_idxs):
            for j, char_idx in enumerate(word_idx):
                context_char_emb[i, j, :] = self.char_embedder.get_embedding(char_idx)
        # context_char_emb = np.max(np.abs(context_char_emb), axis=1)
        # context_char_emb_max = np.max(context_char_emb, axis=1)
        # context_char_emb_min = np.min(context_char_emb, axis=1)
        return context_char_emb.transpose((0, 2, 1))

    def get_question_word_emb(self, idx):
        question_word_idxs = self.datapoints[idx].question_word_idxs
        num_words = self.config.question_limit
        question_word_emb = np.zeros((num_words, self.word_embedder.emb_dim), dtype=np.float32)
        for i, word_idx in enumerate(question_word_idxs):
            question_word_emb[i, :] = self.word_embedder.get_embedding(word_idx)
        return question_word_emb

    def get_question_char_emb(self, idx):
        question_char_idxs = self.datapoints[idx].question_char_idxs
        num_words = self.config.question_limit
        num_chars = self.config.char_limit
        question_char_emb = np.zeros((num_words, num_chars, self.char_embedder.emb_dim), dtype=np.float32)
        for i, word_idx in enumerate(question_char_idxs):
            for j, char_idx in enumerate(word_idx):
                question_char_emb[i, j, :] = self.char_embedder.get_embedding(char_idx)
        # question_char_emb = np.max(np.abs(question_char_emb), axis=1)
        return question_char_emb.transpose((0, 2, 1))

    def get_answer_start_idx(self, idx):
        answer_start_idx = self.datapoints[idx].context_true_answer_start
        return answer_start_idx

    def get_answer_end_idx(self, idx):
        answer_end_idx = self.datapoints[idx].context_true_answer_end
        return answer_end_idx

    def get_id(self, idx):
        id = self.datapoints[idx].id
        return id



