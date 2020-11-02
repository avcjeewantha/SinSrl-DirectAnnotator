from model.data_utils import CoNLLDataset
from model.srl_model import SRLModel
from model.predIdConfig import PredIdConfig
import sys
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6"
def main():
    # create instance of config
    config = PredIdConfig()
    config.layer=int(sys.argv[1])
    config.step=int(sys.argv[2])
    print("iteration: "+str(config.layer))
    print("step: "+str(config.step))

    # build model
    model = SRLModel(config)
    model.build()

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    # train model
    model.train(train, dev, test)
    model.evaluate(test)

if __name__ == "__main__":
    main()
