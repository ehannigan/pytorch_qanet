from collections import Counter

class SquadCounters:
	def __init__(self):
		self.counter_dict = {}
		self.counter_dict['word_counter'] = Counter()
		self.counter_dict['char_counter'] = Counter()
		self.counter_dict['lower_word_counter'] = Counter()
