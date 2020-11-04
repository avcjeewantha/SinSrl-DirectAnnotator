from model.data_utils import CoNLLDataset
from model.srlIdConfig import SrlIdConfig
from model.srl_model import SRLModel
from model.predIdConfig import PredIdConfig
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def model_build_train_eval(config):
    # build model
    model = SRLModel(config)
    model.build()
    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter, config.task)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter, config.task)

    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter, config.task)
    # train model
    model.train(train, dev, test)
    model.evaluate(test)
    model.close_session()

def train_loop(config):
    for layer in range(2,3):
        for step in range(2,3):
            for train_embeddings in [True, False]:
                for nepochs in range(2,4):
                    for dropout in [0.5, 0.5]:
                        for batch_size in range(30, 31):
                            for lr in [0.001, 0.002]:
                                for lr_decay in [0.97, 0.98]:
                                    for model_type in ['slstm', 'lstm']:
                                        config.layer = layer
                                        config.step = step
                                        config.train_embeddings = train_embeddings
                                        config.nepochs = nepochs
                                        config.dropout = dropout
                                        config.batch_size = batch_size
                                        config.lr = lr
                                        config.lr_decay = lr_decay
                                        config.model_type = model_type
                                        model_build_train_eval(config)

def main():
    task = str(sys.argv[1])
    if task == "predId":
        # create instance of config
        config = PredIdConfig()
        train_loop(config)
    elif task == "srlId":
        # create instance of config
        config = SrlIdConfig()
        train_loop(config)


if __name__ == "__main__":
    main()
