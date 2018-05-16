from embed_lib.tokenized_squad import TokenizedSquad
from embed_lib.indexed_squad import IndexedSquad
from embed_lib.squad_pytorch_dataset import SquadPytorchDataset

from qanet_lib.qanet_wrapper import QANetWrapper

from config import Config
from torch.utils.data import DataLoader
from collections import Counter
import string
import re

import os
import json
import sys
import tqdm
from collections import Counter
from sklearn.externals import joblib
import shutil
from helper_functions import evaluate_predictions
import numpy as np
from embed_lib import glove_embed

def get_squad_dataloader(indexed_squad_obj, batch_size, shuffle):

    squad_pytorch_dataset = SquadPytorchDataset(indexed_squad_obj)
    squad_dataloader = DataLoader(squad_pytorch_dataset, batch_size=batch_size, shuffle=shuffle)

    return squad_dataloader


def save_experiment(config):
    experiment_dir = 'experiments/experiment_{}'.format(config.experiment_no)
    shutil.copy2(experiment_dir, 'run_qanet.py')
    shutil.copy2(experiment_dir, '')


def make_dirs(paths):
    directories = paths.split("/")

    for i in range(len(directories)):
        path = ''
        for j in range(i + 1):
            path += directories[j] + '/'
        if not os.path.exists(path):
            os.makedirs(path)

def create_glove_word_embedding(glove_path, vocab):
    num_lines = sum(1 for line in open(glove_path))
    PAD = "--PAD--"
    OOV = "--OOV--"
    embedding_dict = {}
    token2idx = {}
    idx = 0
    idx2token = {}
    idx2vec = {}
    with open(glove_path, 'r', encoding='utf-8') as glove:
        for line in tqdm(glove, total=num_lines):
            array = line.split()
            token = "".join(array[0])  # token in the glove corpus
            vector = list(
                map(float, array[1:]))  # glove vector for the specific token "itm" (can be word/token or character)
            token2idx[token] = idx
            idx += 1
            idx2token[idx] = token
            idx2vec[idx] = vector

def get_tokenized(config, data_file, data_name, path, percentage=1):
    if os.path.exists(path):
        tokenized = joblib.load(path)
    else:
        tokenized = TokenizedSquad(config=config, squad_data=data_file, dataset_name=data_name, percentage=percentage)
        joblib.dump(tokenized, path)
    return tokenized

def get_indexed(config, tokenized, embedding, path):
    if os.path.exists(path):
        indexed = joblib.load(path)
    else:
        indexed = IndexedSquad(config=config, tokenized_squad_obj=tokenized, embedding=embedding)
        joblib.dump(indexed, path)
    return indexed


def main():
    config = Config()
    experiment_dir = config.experiment_dir.format(config.experiment_no)
    checkpoint_dir = os.path.join(experiment_dir, config.checkpoint_dir)
    make_dirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, config.checkpoint_name)
    tmp_data_dir = os.path.join(os.getcwd(), 'tmp_data')
    make_dirs(tmp_data_dir)
    run_type = config.run_type

    #Load data into it's raw format and pass in counters to record which words occur
    open_train_json = open(config.train_data_path)
    train_json = json.load(open_train_json)
    open_dev_json = open(config.dev_data_path)
    squad_dev_data = json.load(open_dev_json)

    if run_type == 'sanity_check':
        #train on small percent of the data
        train_tokenized = get_tokenized(config=config,
                                        data_file=train_json,
                                        data_name='train',
                                        path=os.path.join(tmp_data_dir,'train_tokenized_{}%.sav'.format(config.train_percentage)),
                                        percentage = config.train_percentage)

        dev_tokenized = get_tokenized(config=config,
                                      data_file=train_json,
                                      data_name='validate',
                                      path=os.path.join(tmp_data_dir,'dev_tokenized'))



        # get embedding
        # to get embedding based on a vocab, add parameter tokenized_object_list=[train_tokenized, ...]
        # get_glove_embedding will automatically create vocab counters for you
        embedding = glove_embed.get_glove_embedding(config=config,
                                                    path='embedding_{}.sav'.format(config.embedding_note))


        #index data using embedding
        train_indexed = get_indexed(config=config,
                                    tokenized=train_tokenized,
                                    embedding=embedding,
                                    path=os.path.join(tmp_data_dir,'train_embedding_{}.sav'.format(config.embedding_note)))


        dev_indexed = get_indexed(config=config,
                                  tokenized=dev_tokenized,
                                  embedding=embedding,
                                  path=os.path.join(tmp_data_dir,'dev_embedding_{}.sav'.format(config.embedding_note)))


        trainloader = get_squad_dataloader(indexed_squad_obj=train_indexed, batch_size=config.batch_size, shuffle=True)
        valloader = get_squad_dataloader(indexed_squad_obj=dev_indexed, batch_size=config.batch_size, shuffle=True)
        devloader = get_squad_dataloader(indexed_squad_obj=dev_indexed, batch_size=config.batch_size, shuffle=False)


        qanet_object = QANetWrapper(config, embedding=embedding, checkpoint_path=checkpoint_path)
        if config.train:
            qanet_object.train(trainloader, train_raw=train_tokenized, valloader=valloader, val_raw=dev_tokenized)
        elif config.load:
            qanet_object.load_model(checkpoint_path, config.load_from_epoch_no)

        qanet_object.loss_plotter.plot(os.path.join(experiment_dir, 'loss'))
        qanet_object.f1_plotter.plot(os.path.join(experiment_dir, 'f1'))
        qanet_object.EM_plotter.plot(os.path.join(experiment_dir, 'em'))

        total_loss, prediction_dict = qanet_object.predict(devloader, dev_tokenized)
        joblib.dump(prediction_dict, os.path.join(experiment_dir, os.path.join(tmp_data_dir,'overfit_prediction_dict_dev_100%_epoch1.sav')))
        #pred_answer_dict = joblib.load(os.path.join(experiment_dir, 'overfit_prediction_dict_dev_100%_epoch8.sav'))

        evaluation = evaluate_predictions(squad_dev_data, prediction_dict)
        f1 = evaluation['f1']
        exact_match = evaluation['exact_match']
        print("f1", f1)
        print('exact match', exact_match)




if __name__ == '__main__':
    main()
