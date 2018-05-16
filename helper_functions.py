import torch.nn as nn
from collections import Counter
import string
import re
import sys


def calc_padding(input_size, kernel_size, stride):
    """
    we want to calculate the padding such that y.shape = x.shape for y = layer(x)
    output_height = input_height + 2*padding_height - kernel_height +1 (assuming stride=1)
    output_width = input_width + 2*padding_width - kernel_width + 1 (assuming stride=1)
    we want output_height = input_height and output_width = input_width. Therefore...
    padding_height = (kernel_height - 1)/2
    padding_width = (kernel_width - 1)/2
    """
    # default of pytorch for input_size = (C_in, H_in, W_in)
    if len(input_size) == 3:
        if stride != (1, 1):
            raise ValueError("calc padding only works for stride=(1,1)")
        padding = (0, 0)
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError(
                "the kernel size: {} is incompatible with CnnHighway. With this kernel, the conv output shape will not equal the input shape".format(
                    kernel_size))
        padding_height = int((kernel_size[0] - 1) / 2)
        padding_width = int((kernel_size[1] - 1) / 2)
        return (padding_height, padding_width)
    if len(input_size) == 2:
        if stride != 1:
            raise ValueError("calc padding only works for stride=(1)")
        padding = int((kernel_size - 1) / 2)
        return padding


def exponential_mask(tensor, mask, very_larg_negative_number=-1e20):
    '''
    assign very large negative numbers indices not masked. When the new masked tensor is passed into softmax, it will become essentially 0:
    '''
    # mask shape = [batchsize, CL or QL]
    additive = (1 - mask.float()) * very_larg_negative_number
    # if additive.shape != tensor.shape:
    #     additive.unsqueeze(dim-1)
    return tensor + additive


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



def LayerDropout(layer_dropout=0, total_layers=0):
    #return nn.Dropout(1-(1/total_layers)*(1-layer_dropout))
    return nn.Dropout(layer_dropout)


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_predictions(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                #print("qa[id]", qa['id'])
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                #print('prediction', prediction)
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print()

    return {'exact_match': exact_match, 'f1': f1}


