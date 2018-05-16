from tqdm import tqdm
import numpy as np
from embed_lib.squad_counters import SquadCounters
import json
import os

def get_glove_embedding(config, path, tokenized_object_list=None):

    if os.path.exists(config.embedding_path):
        embedding = json.load(config.embedding_path)
    else:
        if tokenized_object_list:
            vocab = SquadCounters()
            for tokenized in tokenized_object_list:
                vocab.add_to_vocab(tokenized_squad=tokenized)
            word2idx_dict, word_emb_weights = get_vocab_based_glove_embedding(glove_path=config.glove_word_embedding_path,
                                                                              vocab=vocab['word_counter'])
            char2idx_dict, char_emb_weights = get_vocab_based_glove_embedding(glove_path=config.glove_char_embedding_path,
                                                                              emb_dim=vocab['char_counter'])
        else:
            word2idx_dict, word_emb_weights = get_standardized_glove_embedding(glove_path=config.glove_word_embedding_path,
                                                                               emb_dim=config.glove_word_dim,
                                                                               vocab_size=config.word_vocab_size)
            char2idx_dict, char_emb_weights = get_standardized_glove_embedding(glove_path=config.glove_char_embedding_path,
                                                                               emb_dim=config.glove_char_dim,
                                                                               vocab_size=config.char_vocab_size)

        embedding = {'word2idx_dict': word2idx_dict, 'word_emb_weights': word_emb_weights,
                     'char2idx_dict': char2idx_dict, 'char_emb_weights': char_emb_weights}

        json.dump(embedding, path)

    return embedding


def get_standardized_glove_embedding(glove_path, emb_dim, vocab_size=40000):
    # https://damienpontifex.com/2017/10/27/using-pre-trained-glove-embeddings-in-tensorflow/
    PAD = 0
    OOV = 1
    num_lines = sum(1 for line in open(glove_path))
    token2idx = {}
    embedding_weights = np.zeros((vocab_size, emb_dim))
    with open(glove_path, 'r', encoding='utf-8') as glove:
        idx = 2 #leave first two indices for padding(0) and unk(1)
        for line in tqdm(glove, total=num_lines):

            array = line.split()
            token = "".join(array[0])  # token in the glove corpus
            vector = np.asarray(array[1:], dtype=np.float32)  # glove vector for the specific token "itm" (can be word/token or character)
            token2idx[token] = idx
            embedding_weights[idx] = vector
            idx += 1
            if idx >= vocab_size:
                # we will only use the top vocab_size words in glove
                break
        token2idx['PAD'] = PAD
        token2idx['OOV'] = OOV
        embedding_weights[PAD] = np.random.randn(emb_dim)
        embedding_weights[OOV] = np.random.randn(emb_dim)
    return token2idx, embedding_weights


def get_vocab_based_glove_embedding(vocab, glove_path):
    # https://github.com/NLPLearn/QANet/blob/master/prepro.py
    num_lines = sum(1 for line in open(glove_path))
    embedding_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as glove:
        for line in tqdm(glove, total=num_lines ):
            array = line.split()
            itm = "".join(array[0])
            vector = list(map(float, array[1:]))
            if itm in vocab:
                embedding_dict[itm] = vector
            elif itm.capitalize() in vocab:
                embedding_dict[itm.capitalize()] = vector
            elif itm.lower() in vocab:
                embedding_dict[itm.lower()] = vector
            elif itm.upper() in vocab:
                embedding_dict[itm.upper()] = vector
    PAD = 1
    OOV = 0
    # token2idx_dict[token] = idx of all possible embeddings
    token2idx_dict = {token: int(idx) for idx, token in enumerate(embedding_dict.keys(),2)}  # start enumerate at 2 so that we have room for null and oov
    token2idx_dict['OOV'] = OOV  # out of vocabulary
    token2idx_dict['PAD'] = PAD  # padding
    embedding_size = len(next(iter(embedding_dict.values())))
    embedding_dict['PAD'] = [list(np.random.randn(embedding_size))]
    embedding_dict['OOV'] = [list(np.random.randn(embedding_size))]
    idx2emb_dict = {int(idx): embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = np.asarray([idx2emb_dict[idx] for idx in range(len(idx2emb_dict))])

    return token2idx_dict, emb_mat