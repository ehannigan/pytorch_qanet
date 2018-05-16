class Config:
    def __init__(self):
        self.run_type = 'sanity_check'
        self.train = True
        self.train_percentage = .05
        self.load = False
        self.experiment_no = 0
        self.load_from_epoch_no = 0
        self.experiment_dir = 'experiments/experiment_{}/'
        self.checkpoint_dir = 'model_checkpoints/'
        self.checkpoint_name = 'checkpoint_epoch_{}'


        self.char_limit = 16
        self.context_limit = 400
        self.question_limit = 50
        self.answer_limit = 30
        self.glove_word_size = int(2.2e6)
        self.d_model = 96

        #training
        self.batch_size = 32
        self.max_val_batches = 10
        self.num_epochs = 10
        self.print_freq = 100
        self.learning_rate = 0.001
        self.num_learning_rate_warm_up_steps = 1000
        self.optimizer_name = 'adam'
        self.criterion_name = 'cross_entropy_loss'

        self.dev_data_path = 'data_files/squad_datasets/dev-v1.1.json'
        self.train_data_path = 'data_files/squad_datasets/train-v1.1.json'

        self.glove_word_embedding_path = 'data_files/glove_txt/glove.6B.200d.txt'
        self.glove_word_dim = 200
        self.glove_char_embedding_path = 'data_files/glove_txt/glove.840B.300d-char.txt'
        self.glove_char_dim = 300
        self.word_vocab_size = 40000
        self.char_vocab_size = 94
        self.embedding_note = 'word_vocab_40000'

        self.train_val_plotter_path = 'train_val_plotter.sav'

        self.cuda_flag = True



        #dropout
        self.word_emb_dropout = .1
        self.char_emb_dropout = .05
        self.trilinear_dropout = 0
        self.highway_dropout = 0
        self.layer_dropout = 0
        self.general_dropout = .1

        #masking
        self.self_attention_mask = False
        self.c2q_mask = True
        self.pred_mask = True

        #highway variables
        self.hw_layers = 2
        self.hw_stride = 1
        self.hw_kernel = 3
        self.highway_dropout = 0


        #context query attention
        self.cqa_dropout = 0

        #embedding block variables
        self.num_emb_blocks = 1
        self.num_emb_conv = 4
        self.emb_kernel = 7
        self.emb_depthwise = True
        self.emb_num_heads = 8

        #model block variables
        self.num_mod_blocks = 2
        self.num_mod_conv = 2
        self.mod_kernel = 5
        self.mod_depthwise = True
        self.mod_num_heads = 8


        