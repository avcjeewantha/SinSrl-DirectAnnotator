import os
from .general_utils import get_logger
from .data_utils import get_trimmed_fasttext_vectors, load_vocab, \
    get_processing_word


class SrlIdConfig():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs
        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None
        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        # create instance of logger
        self.logger = get_logger(self.path_log)
        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings
        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed Fasttext
        vectors)
        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag = get_processing_word(self.vocab_tags,
                                                  lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_fasttext_vectors(self.filename_trimmed)
                           if self.use_pretrained else None)

    # general config
    dir_output = "results/test/srlIdData/"
    dir_model = dir_output + "model.weights/"
    path_log = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # fasttext files
    filename_fasttext = "data/fasttext_{}_nsw.w2v".format(dim_word)
    # trimmed embeddings (created from fasttext_filename with build_data.py)
    filename_trimmed = "data/srlIdData/fasttext_{}_nsw.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_train = "data/srlIdData/train.txt"  # test
    filename_dev = "data/srlIdData/dev.txt"
    filename_test = "data/srlIdData/test.txt"

    max_iter = None  # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/srlIdData/words.txt"
    filename_tags = "data/srlIdData/tags.txt"
    filename_chars = "data/srlIdData/chars.txt"

    # training
    layer = 10  # iteration
    step = 2
    train_embeddings = False
    nepochs = 4  # 100
    dropout = 0.5
    batch_size = 30
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.97
    clip = 3  # if negative, no clipping
    nepoch_no_imprv = 100

    # model hyperparameters
    hidden_size_char = 150  # lstm on chars
    hidden_size_lstm = 300  # lstm on word embeddings
    hidden_size_sum = 600
    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU
    char_use_mlstm = False
    random_initialize = True
    task = "srlId"  # srlId
    model_type = 'slstm'
