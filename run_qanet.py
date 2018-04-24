from embed_lib.squad_raw import SquadRaw
from embed_lib.squad_emb import SquadEmb
from embed_lib.glove_embedding import GloveEmbedding
from embed_lib.squad_pytorch_dataset import SquadPytorchDataset
from qanet_lib.qanet import QANet
from qanet_lib.qanet_wrapper import QANetWrapper
from config import Config
import json
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from sklearn.externals import joblib

from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def get_squad_dataloader(squad_emb, batch_size, shuffle):

    squad_pytorch_dataset = SquadPytorchDataset(squad_emb)
    squad_dataloader = DataLoader(squad_pytorch_dataset, batch_size=batch_size, shuffle=shuffle)

    return squad_dataloader


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def main():

    open_dev_json = open('dev-v1.1.json')
    open_train_json = open('train-v1.1.json')


    squad_dev_data = json.load(open_dev_json)
    squad_train_data = json.load(open_train_json)

    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    config = Config()


    train_raw = SquadRaw(config, squad_train_data, word_counter, char_counter, lower_word_counter, percentage=.1)
    dev_raw = SquadRaw(config, squad_dev_data, word_counter, char_counter, lower_word_counter)
    #joblib.dump(dev_raw, 'dev_raw.sav')
    #dev_raw = joblib.load('dev_raw.sav')




    glove_word = GloveEmbedding(word_counter, config.glove_word_embedding_path, config.glove_word_size)
    glove_char = GloveEmbedding(char_counter, config.glove_char_embedding_path, config.glove_char_size)
    # joblib.dump(glove_word, 'glove_word.sav')
    # joblib.dump(glove_char, 'glove_char.sav')
    #glove_word = joblib.load('glove_word.sav')
    #glove_char = joblib.load('glove_char.sav')
    #train_emb = SquadEmb(train_raw, glove_word, glove_char, config)
    dev_emb = SquadEmb(dev_raw, glove_word, glove_char, config)
    #joblib.dump(dev_emb, 'dev_emb.sav')

    #trainloader = get_squad_dataloader(train_emb, batch_size=config.batch_size, shuffle=True)
    devloader = get_squad_dataloader(dev_emb, batch_size=config.batch_size, shuffle=False)
    print('created devloader')

    qanet_object = QANetWrapper(config)
    qanet_object.load_model('checkpoint.pth')
    print('created QANetWRapper')
    #qanet_object.train(trainloader)
    total_loss, pred_answer_dict = qanet_object.test(devloader, dev_raw)
    print("dev loss: {}".format(total_loss))
    evaluation = evaluate(squad_dev_data, pred_answer_dict)
    f1 = evaluation['f1']
    print("f1", f1)



if __name__ == '__main__':
    main()
