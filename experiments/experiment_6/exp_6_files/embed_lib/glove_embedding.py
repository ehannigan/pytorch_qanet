from tqdm import tqdm
import numpy as np
class GloveEmbedding:
    def __init__(self, counter, glove_embedding_path, corpus_size):
        self.glove_embedding_path = glove_embedding_path
        self.corpus_size = corpus_size
        self.counter = counter

        self.embedding_dict = self.__get_embedding_dict()
        self.emb_mat, self.token2idx_dict = self.__get_embedding_mat()
        self.get_embeddings()

    def get_embeddings(self):
        return self.embedding_dict, self.emb_mat, self.token2idx_dict

    def __get_embedding_dict(self):
        """
        creates glove word embedding. embedding_dict[word]
        returns the glove vector
        """
        #print("getting glove embedding from {}".format(self.glove_embedding_path))
        embedding_dict = {}
        with open(self.glove_embedding_path, 'r', encoding='utf-8') as glove:
            for line in tqdm(glove, total=self.corpus_size):
                array = line.split()
                itm = "".join(array[0])
                vector = list(map(float, array[1:]))
                if itm in self.counter:
                    embedding_dict[itm] = vector
                elif itm.capitalize() in self.counter:
                    embedding_dict[itm.capitalize()] = vector
                elif itm.lower() in self.counter:
                    embedding_dict[itm.lower()] = vector
                elif itm.upper() in self.counter:
                    embedding_dict[itm.upper()] = vector
        return embedding_dict


    def __get_embedding_mat(self):
        # embedding_dict[token] = embedding
        PAD = "--PAD--"
        OOV = "--OOV--"
        # token2idx_dict[token] = idx of all possible embeddings
        token2idx_dict = {token: int(idx) for idx, token in
                          enumerate(self.embedding_dict.keys(), 2)}  # start enumerate at 2 so that we have room for null and oov
        token2idx_dict[OOV] = 1  # out of vocabulary
        token2idx_dict[PAD] = 0  # padding
        embedding_size = len(next(iter(self.embedding_dict.values())))
        self.embedding_dict[PAD] = [0 for i in range(embedding_size)]
        self.embedding_dict[OOV] = [0 for i in range(embedding_size)]
        idx2emb_dict = {int(idx): self.embedding_dict[token] for token, idx in token2idx_dict.items()}
        emb_mat = np.asarray([idx2emb_dict[idx] for idx in range(len(idx2emb_dict))])

        return emb_mat, token2idx_dict