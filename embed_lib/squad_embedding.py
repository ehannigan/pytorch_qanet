class squad_embedding():
    def get_embeddings(counter, glove_embedding_path):
        embedding_dict = get_embedding_dict(counter, glove_embedding_path)
        emb_mat, token2idx_dict = get_embedding_mat(get_embedding_mat)

        return embedding_dict, emb_mat, token2idx_dict


    def get_embedding_dict(counter, glove_embedding_path, limit=0):
        """
        creates glove word embedding. embedding_dict[word]
        returns the glove vector
        """
        embedding_dict = {}
        glove_file = 
        with open(glove_embedding_path, 'r', encoding='utf-8') as glove:
            for line in tqdm(glove, total=vocab_size):
                array = line.split()
                word = "".join(array[0])
                vector = list(map(float, array[1:]))
                if itm in counter and counter[word]>limit:
                    embedding_dict[itm] = vector
                elif word.capitalize() in counter:
                    embedding_dict[itm.capitalize()] = vector
                elif word.lower() in counter:
                    embedding_dict[itm.lower()] = vector
                elif word.upper() in counter:
                    embedding_dict[itm.upper()] = vector
        return embedding_dict

    def get_embedding_mat(embedding_dict):
        NULL = "--NULL--"
        OOV = "--OOV--"
        token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)} #start enumerate at 2 so that we have room for null and oov
        token2idx_dict['OOV'] = 1 # out of vocabulary
        token2idx_dict['NULL'] = 0 # padding 
        embedding_size = len(next(iter(embedding_dict.values())))
        embedding_dict[NULL] = [0 for i in range(embedding_size)]
        embedding_dict[OOV] = [0 for i in range(embedding_size)]

        idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
        emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

        return emb_mat, token2idx_dict