import sys
from model.predIdConfig import PredIdConfig
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_fasttext_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_fasttext_vectors, get_processing_word
from model.srlIdConfig import SrlIdConfig


def process_data(config, task):
    processing_word = get_processing_word()

    # Generators (Here it is only initialized)
    dev = CoNLLDataset(config.filename_dev, processing_word, task=task)
    test = CoNLLDataset(config.filename_test, processing_word, task=task)
    train = CoNLLDataset(config.filename_train, processing_word, task=task)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test]) #set of words, set of tags
    vocab_fasttext = get_fasttext_vocab(vocab_words, config.filename_fasttext) #set of vocab_words which can be represented by vectors

    vocab = vocab_fasttext
    vocab.add(UNK) #$UNK$
    vocab.add(NUM) #$NUM$

    # Save vocab
    write_vocab(vocab, config.filename_words) #write words in vocan to words.txt file
    write_vocab(vocab_tags, config.filename_tags) #write tags in vocab_tags to tags.txt file

    # Trim Fasttext Vectors
    vocab = load_vocab(config.filename_words) # A dictionary- {key-word in words.txt, value- line no. of the word in words.txt}
    export_trimmed_fasttext_vectors(vocab, config.filename_fasttext,
                                    config.filename_trimmed, config.dim_word) #Save several arrays into a single file in compressed .npz format

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train, task=task)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars) #write characters in train words to chars.txt file


def main(task=None):
    """Procedure to build data
    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant Fasttext vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.
    """
    if(task == None):
        task = str(sys.argv[1])
    if task == "predId":
        # get config and processing of words
        config = PredIdConfig(load=False)
        process_data(config, task)
    elif task == "srlId":
        # get config and processing of words
        config = SrlIdConfig(load=False)
        process_data(config, task)


if __name__ == "__main__":
    main()