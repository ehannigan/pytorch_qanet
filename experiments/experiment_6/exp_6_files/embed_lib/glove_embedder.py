from tqdm import tqdm
import numpy as np
from sklearn.externals import joblib
class GloveEmbedder:
    def __init__(self, counter, glove_embedding_path, corpus_size):
        self.glove_embedding_path = glove_embedding_path
        self.corpus_size = corpus_size
        self.PAD = "--PAD--"
        self.OOV = "--OOV--"
        embedding_dict = self.__get_embedding_dict(counter)
        self.token2idx_dict = self.__get_token2idx_dict(embedding_dict)
        self.emb_mat = self.__get_embedding_mat(embedding_dict)
        self.emb_dim = self.__get_emb_dim()

    def __get_emb_dim(self):
        emb_dim = self.emb_mat.shape[1]
        return emb_dim

    def __get_embedding_dict(self, counter):
        """
        embedding_dict[token] = embedding
        creates glove word embedding. embedding_dict[token] = glove_vector
        returns the glove vector
        """
        #print("getting glove embedding from {}".format(self.glove_embedding_path))
        embedding_dict = {}
        with open(self.glove_embedding_path, 'r', encoding='utf-8') as glove:
            for line in tqdm(glove, total=self.corpus_size):
                array = line.split()
                itm = "".join(array[0])  # token in the glove corpus
                vector = list(map(float, array[1:]))  # glove vector for the specific token "itm" (can be word/token or character)
                if itm in counter:
                    embedding_dict[itm] = vector
                elif itm.capitalize() in counter:
                    embedding_dict[itm.capitalize()] = vector
                elif itm.lower() in counter:
                    embedding_dict[itm.lower()] = vector
                elif itm.upper() in counter:
                    embedding_dict[itm.upper()] = vector

        # create pad and out of vocabulary vector
        embedding_size = len(next(iter(embedding_dict.values())))
        embedding_dict[self.PAD] = [0 for i in range(embedding_size)]
        embedding_dict[self.OOV] = [.5 for i in range(embedding_size)]
        return embedding_dict

    def __get_token2idx_dict(self, embedding_dict):
        # token2idx_dict[token] = idx of tokens embedding int he embedding dict
        # enumerate(self.embedding_dict.keys()) will return all of the tokens/words/chars in the corpus along with its index in the dict
        # token2idx_dict[token] will be the index of the word in the embedding dictionary (basically an arbitrary index for convenience)
        token2idx_dict = {token: int(idx) for idx, token in
                          enumerate(embedding_dict.keys(), 2)}  # start enumerate at 2 so that we have room for null and oov (out of vocabulary)
        token2idx_dict[self.OOV] = 1  # out of vocabulary
        token2idx_dict[self.PAD] = 0  # padding
        return token2idx_dict

    def __get_embedding_mat(self, embedding_dict):
        idx2emb_dict = {int(idx): embedding_dict[token] for token, idx in self.token2idx_dict.items()}
        emb_mat = np.asarray([idx2emb_dict[idx] for idx in range(len(idx2emb_dict))])  #emb_mat[idx] = embedding vector
        # from any token/char we can use token2idx_dict[token] to get its index in the emb_mat, and from there get its glove embedding\
        return emb_mat

    def get_embedding(self, idx):
        emb = self.emb_mat[idx, :]
        return emb

    def get_indexing(self, token):
        if token in self.token2idx_dict:
            return self.token2idx_dict[token]
        return 1  # Out of vocabulary



