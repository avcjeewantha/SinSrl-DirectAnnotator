from model.data_utils import CoNLLDataset
from model.srl_model import SRLModel
from model.predIdConfig import PredIdConfig
import sys


def main():
    # create instance of config
    config = PredIdConfig()
    config.layer = int(sys.argv[1])
    config.step = int(sys.argv[2])
    print("iteration: " + str(config.layer))
    print("step: " + str(config.step))

    # build model
    model = SRLModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)

    # evaluate
    model.evaluate(test)


if __name__ == "__main__":
    main()
