class Config:
    def __init__(self):
        self.char_limit = 16
        self.context_limit = 400
        self.question_limit = 50
        self.answer_limit = 30
        self.glove_word_size = int(2.2e6)
        self.glove_word_dim = 200
        self.glove_char_size = 94
        self.glove_char_dim = 300

        self.batch_size = 100
        self.num_epochs = 1
        self.shuffle = True
        self.print_freq = 100

        self.glove_word_embedding_path = 'glove.6B.200d.txt'
        self.glove_char_embedding_path = 'glove.840B.300d-char.txt'

        self.cuda_flag = True

        self.hidden_size = 16

        #highway variables
        self.hw_layers = 2
        self.hw_stride = 1
        self.hw_kernel = 3
        self.learning_rate = .0001

        #embedding block variables
        self.num_emb_blocks = 1
        self.num_emb_conv = 4
        self.emb_kernel = 7
        self.emb_depthwise = False

        #model block variables
        self.num_mod_blocks = 7
        self.num_mod_conv = 2
        self.mod_kernel = 5
        self.mod_depthwise = True


        