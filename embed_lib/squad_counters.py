from collections import Counter


class SquadCounters:
    def __init__(self):
        self.counter_dict = {}
        self.counter_dict['word_counter'] = Counter()
        self.counter_dict['char_counter'] = Counter()
        self.counter_dict['lower_word_counter'] = Counter()

    def add_to_vocab(self, tokenized_squad):
        for datapoint in tokenized_squad.datapoint_dict.items():
            for word in datapoint.context_tokens:
                self.counter_dict['word_counter'][word] += 1
                self.counter_dict['lower_word_counter'][word.lower()] += 1
                for char in word:
                    self.counter_dict['char_counter'][char] += 1

            for word in datapoint.question_tokens:
                self.counter_dict['word_counter'][word] += 1
                self.counter_dict['lower_word_counter'][word] += 1
                for char in word:
                    self.counter_dict['char_counter'][char] += 1
