from embed_lib.squad_raw import SquadRaw
from embed_lib.squad_emb import SquadEmb
from embed_lib.glove_embedder import GloveEmbedder
from embed_lib.squad_pytorch_dataset import SquadPytorchDataset
from qanet_lib.qanet_wrapper import QANetWrapper
from squad_counters import SquadCounters
from config import Config
from torch.utils.data import DataLoader
import json
import sys
import argparse

def get_squad_dataloader(squad_emb, batch_size, shuffle):

    squad_pytorch_dataset = SquadPytorchDataset(squad_emb)
    squad_dataloader = DataLoader(squad_pytorch_dataset, batch_size=batch_size, shuffle=shuffle)

    return squad_dataloader


def main(testing_json_path, output_json_path):
    config = Config()
    squad_test_data = json.load(testing_json_path)
    qanet_object = QANetWrapper(config)
    qanet_object.load_model(config.checkpoint_load_path)
    counters = SquadCounters()
    test_raw = SquadRaw(config, squad_test_data, counters, 'test')
    glove_word_embedder = GloveEmbedder(counters.counter_dict['word_counter'], config.glove_word_embedding_path, config.glove_word_size)
    glove_char_embedder = GloveEmbedder(counters.counter_dict['char_counter'], config.glove_char_embedding_path, config.glove_char_size)
    test_emb = SquadEmb(config, test_raw, glove_word_embedder, glove_char_embedder)
    testloader = get_squad_dataloader(test_emb, batch_size=config.batch_size, shuffle=False)
    total_loss, pred_answer_dict = qanet_object.predict(testloader, test_raw)
    with open(output_json_path, 'w') as outfile:
        json.dump(pred_answer_dict, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='enter input json file and output json file')
    testing_json_path = sys.argv[1]
    output_json_path = sys.argv[2]
    main(testing_json_path, output_json_path)
