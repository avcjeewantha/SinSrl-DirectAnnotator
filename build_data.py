from model.predIdConfig import PredIdConfig
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_fasttext_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_fasttext_vectors, get_processing_word

def main():
    """Procedure to build data
    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant Fasttext vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.

    Args:
        config: (instance of PredIdConfig) has attributes like hyper-params...
    """
    # get config and processing of words
    config = PredIdConfig(load=False)
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_fasttext = get_fasttext_vocab(vocab_words,config.filename_fasttext)

    vocab = vocab_fasttext
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim Fasttext Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_fasttext_vectors(vocab, config.filename_fasttext,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)

if __name__ == "__main__":
    main()