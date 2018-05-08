from embed_lib.squad_raw import SquadRaw
from embed_lib.squad_emb import SquadEmb
from embed_lib.glove_embedder import GloveEmbedder
from embed_lib.squad_pytorch_dataset import SquadPytorchDataset
from embed_lib.squad_counters import SquadCounters
from qanet_lib.qanet_wrapper import QANetWrapper

from config import Config
from torch.utils.data import DataLoader
from collections import Counter
import string
import re
import json
import os
import sys
from sklearn.externals import joblib
import shutil
from helper_functions import evaluate_predictions




def get_squad_dataloader(squad_emb, batch_size, shuffle):

    squad_pytorch_dataset = SquadPytorchDataset(squad_emb)
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


def main():
    config = Config()
    experiment_dir = config.experiment_dir.format(config.experiment_no)
    checkpoint_dir = os.path.join(experiment_dir, config.checkpoint_dir)
    make_dirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, config.checkpoint_name)
    run_type = 'new_run'



    if run_type == 'load':
        dev_raw = joblib.load('dev_raw.sav')
        print('raw dev loaded')
        train_emb = joblib.load('train_embedding_20%.sav')
        print('train emb loaded')
        dev_emb = joblib.load('dev_embedding.sav')
        print('dev emb loaded')
        open_dev_json = open(config.dev_data_path)
        squad_dev_data = json.load(open_dev_json)

    elif run_type == 'new_run':
        counters = SquadCounters()
        #Load data into it's raw format and pass in counters to record which words occur
        open_train_json = open(config.train_data_path)
        squad_train_data = json.load(open_train_json)
        #train_raw = SquadRaw(config, squad_train_data, counters, 'train', percentage=1)
        #joblib.dump(train_raw, 'train_raw_100%.sav')
        train_raw = joblib.load('train_raw_100%.sav')

        open_dev_json = open(config.dev_data_path)
        squad_dev_data = json.load(open_dev_json)
        #dev_raw = SquadRaw(config, squad_dev_data, counters, 'validate')
        #joblib.dump(dev_raw, 'dev_raw.sav')
        dev_raw = joblib.load('dev_raw.sav')

        #glove_word_embedder = GloveEmbedder(counters.counter_dict['word_counter'], config.glove_word_embedding_path, config.glove_word_size)
        #glove_char_embedder = GloveEmbedder(counters.counter_dict['char_counter'], config.glove_char_embedding_path, config.glove_char_size)
        #joblib.dump(glove_word_embedder, 'glove_word_embedder.sav')
        #joblib.dump(glove_char_embedder, 'glove_char_embedder.sav')

        #train_emb = SquadEmb(config, train_raw, glove_word_embedder, glove_char_embedder)
        #joblib.dump(train_emb, 'train_embedding_100%.sav')
        train_emb = joblib.load('train_embedding_100%.sav')
        #dev_emb = SquadEmb(config, dev_raw, glove_word_embedder, glove_char_embedder)
        #joblib.dump(dev_emb, 'dev_embedding.sav')
        dev_emb = joblib.load('dev_embedding.sav')

        trainloader = get_squad_dataloader(train_emb, batch_size=config.batch_size, shuffle=True)
        valloader = get_squad_dataloader(dev_emb, batch_size=config.batch_size, shuffle=True)
        devloader = get_squad_dataloader(dev_emb, batch_size=config.batch_size, shuffle=False)
        qanet_object = QANetWrapper(config, checkpoint_path)
        qanet_object.train(trainloader, train_raw=train_raw, valloader=valloader, val_raw=dev_raw)
        #qanet_object.load_model(checkpoint_path, config.load_from_epoch_no)
        # qanet_object.loss_plotter.plot(os.path.join(experiment_dir, 'overfit_loss'))
        # qanet_object.f1_plotter.plot(os.path.join(experiment_dir, 'overfit_f1'))
        # qanet_object.EM_plotter.plot(os.path.join(experiment_dir, 'overfit_em'))

        total_loss, prediction_dict = qanet_object.predict(devloader, dev_raw)
        joblib.dump(prediction_dict, os.path.join(experiment_dir, 'overfit_prediction_dict_dev_100%_epoch1.sav'))
        #pred_answer_dict = joblib.load(os.path.join(experiment_dir, 'overfit_prediction_dict_dev_100%_epoch8.sav'))
        #print("dev loss: {}".format(total_loss))
        evaluation = evaluate_predictions(squad_dev_data, prediction_dict)
        f1 = evaluation['f1']
        exact_match = evaluation['exact_match']
        print("f1", f1)
        print('exact match', exact_match)

    elif run_type == 'overfit':
        counters = SquadCounters()
        # Load data into it's raw format and pass in counters to record which words occur
        open_train_json = open(config.train_data_path)
        squad_train_data = json.load(open_train_json)
        #train_raw = SquadRaw(config, squad_train_data, counters, 'train', percentage=.007)
        train_raw = joblib.load('train_raw_07%.sav')
        #joblib.dump(train_raw, 'train_raw_07%.sav')

        #test_raw = SquadRaw(config, squad_train_data, counters, 'validate')
        test_raw = joblib.load('train_raw_07%.sav')
        # glove_word_embedder = GloveEmbedder(counters.counter_dict['word_counter'], config.glove_word_embedding_path,
        #                                     config.glove_word_size)
        # glove_char_embedder = GloveEmbedder(counters.counter_dict['char_counter'], config.glove_char_embedding_path,
        #                                     config.glove_char_size)
        # joblib.dump(glove_word_embedder, 'glove_word_embedder_07%.sav')
        # joblib.dump(glove_char_embedder, 'glove_char_embedder_07%.sav')

        #train_emb = SquadEmb(config, train_raw, glove_word_embedder, glove_char_embedder)
        #joblib.dump(train_emb, 'train_embedding_07%.sav')
        train_emb = joblib.load('train_embedding_07%.sav')

        #test_emb = SquadEmb(config, test_raw, glove_word_embedder, glove_char_embedder)
        test_emb = joblib.load('train_embedding_07%.sav')

        trainloader = get_squad_dataloader(train_emb, batch_size=config.batch_size, shuffle=True)
        valloader = get_squad_dataloader(test_emb, batch_size=config.batch_size, shuffle=True)
        devloader = get_squad_dataloader(test_emb, batch_size=config.batch_size, shuffle=False)
        qanet_object = QANetWrapper(config, checkpoint_path)
        qanet_object.train(trainloader, train_raw=train_raw, valloader=valloader, val_raw=test_raw)
        #qanet_object.train_val_plotter.plot_train_val(os.path.join(experiment_dir, 'overfit_train_val_plot'))
        #qanet_object.load_model(checkpoint_path, config.load_from_epoch_no)
        total_loss, prediction_dict = qanet_object.predict(devloader, test_raw)
        joblib.dump(prediction_dict, os.path.join(experiment_dir, 'overfit_prediction_dict.sav'))
        print("dev loss: {}".format(total_loss))
        evaluation = evaluate_predictions(squad_train_data, prediction_dict)
        f1 = evaluation['f1']
        print("f1", f1)
    #
    # trainloader = get_squad_dataloader(train_emb, batch_size=config.batch_size, shuffle=True)
    # devloader = get_squad_dataloader(dev_emb, batch_size=config.batch_size, shuffle=False)
    #
    # qanet_object = QANetWrapper(config, checkpoint_path)
    # #qanet_object.train(trainloader)
    # #qanet_object.train_val_plotter.plot_train_val(os.path.join(experiment_dir, 'train_val_plot'))
    # qanet_object.load_model(checkpoint_path, config.load_from_epoch_no)
    # total_loss, pred_answer_dict = qanet_object.predict(devloader, dev_raw)
    # joblib.dump(pred_answer_dict, os.path.join(experiment_dir, 'dev_prediction_dict.sav'))
    #
    #
    # print("dev loss: {}".format(total_loss))
    # evaluation = evaluate(squad_dev_data, pred_answer_dict)
    # f1 = evaluation['f1']
    # print("f1", f1)



if __name__ == '__main__':
    main()
