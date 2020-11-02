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


def main():
    task = str(sys.argv[1])
    if task == "predId":
        # create instance of config
        config = PredIdConfig()
        model_build_train_eval(config)
    elif task == "srlId":
        # create instance of config
        config = SrlIdConfig()
        model_build_train_eval(config)


if __name__ == "__main__":
    main()
