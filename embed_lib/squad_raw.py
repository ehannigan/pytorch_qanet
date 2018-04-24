from embed_lib.raw_datapoint import RawDatapoint
import re
import nltk

from tqdm import tqdm
class SquadRaw:

    def __init__(self, config, squad_data, word_counter, char_counter, lower_word_counter, percentage=1):
        self.datapoint_dict = {}
        self.config = config
        self.__load_squad_data(squad_data, word_counter, char_counter, lower_word_counter, percentage=percentage)

    def __process_tokens(self, temp_tokens):
        tokens = []
        for token in temp_tokens:
            flag = False
            l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
            # \u2013 is en-dash. Used for number to nubmer
            # l = ("-", "\u2212", "\u2014", "\u2013")
            # l = ("\u2013",)
            tokens.extend(re.split("([{}])".format("".join(l)), token))
            #print('tokens', tokens)
        return tokens

    def __get_context_spans(self, context, context_tokens):
        # span is the tuple (word_start_char, word_end_char) that gives where each word
        # starts and ends within a context
        current_char = 0
        context_spans = []
        for token in context_tokens:
            current_char = context.find(token, current_char)
            if current_char < 0: #aka the token doesn't exist after current_char
                raise Exception()
            context_spans.append([current_char, current_char+len(token)])
            current_char += len(token)
        return context_spans

    def __load_squad_data(self, squad_data, word_counter, char_counter, lower_word_counter, percentage):
        # parses squad data into context, questions, and answers
        start_ai = 0
        stop_ai = int(round(len(squad_data['data']) * percentage))
        for article_idx, article in enumerate(tqdm(squad_data['data'][start_ai:stop_ai])):
            total_question_count = 0
            
            for para_idx, para in enumerate(article['paragraphs']):
                context = para['context'].replace("''", '" ').replace("``", '" ')
                context_tokens = [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(context)]
                context_chars = [list(word) for word in context_tokens]
                context_spans = self.__get_context_spans(context, context_tokens)
                
                for word in context_tokens:
                    word_counter[word] += len(para['qas'])
                    lower_word_counter[word.lower()] += len(para['qas'])
                    for char in word:
                        char_counter[char] += len(para['qas'])

                for qa in para['qas']:
                    total_question_count += 1
                    question = qa['question']
                    question_tokens = nltk.word_tokenize(question)
                    question_chars = [list(word) for word in question_tokens]

                    answer_list = []
                    for word in question_tokens:
                        word_counter[word] += 1
                        lower_word_counter[word] += 1
                        for char in word:
                            char_counter[char] += 1
                    answer_start_idxs = []
                    answer_end_idxs = []
                    answer_texts = []
                    for answ_obj in qa['answers']:
                        answer_text = answ_obj['text']
                        answer_texts.append(answer_text)
                        answer_start_char_idx = answ_obj['answer_start']
                        answer_stop_char_idx = answer_start_char_idx + len(answer_text)
                        answer_span = []

                        for word_idx, word_span in enumerate(context_spans):
                            if not (answer_stop_char_idx <= word_span[0] or answer_start_char_idx >= word_span[1]):
                                answer_span.append(word_idx)
                        answer_start_idx, answer_end_idx = answer_span[0], answer_span[-1]
                        
                        # hold the index of the word/token of where the answer starts and stops
                        answer_start_idxs.append(answer_start_idx)
                        answer_end_idxs.append(answer_end_idx)
                    
                    example = RawDatapoint()
                    example.context = context
                    example.context_tokens = context_tokens  # process tokens
                    example.context_chars = context_chars
                    example.context_spans = context_spans

                    example.question_tokens = question_tokens
                    example.question_chars = question_chars
                    example.question_id = total_question_count

                    example.answer_texts = answer_texts
                    example.answer_start_idxs = answer_start_idxs
                    example.answer_end_idxs = answer_end_idxs
                    example.uuid = qa['id']
                    self.datapoint_dict[str(qa['id'])] = example






