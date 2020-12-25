from model.data_utils import CoNLLDataset
from model.srlIdConfig import SrlIdConfig
from model.srl_model import SRLModel
from model.predIdConfig import PredIdConfig
import build_data
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

finalResults = []
allParams = []

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
    params = model.train(train, dev, test)
    results = model.evaluate(test)
    finalResults.append(results)
    allParams.append(params)
    model.close_session()

# For the re training process
def model_build_retrain_eval(config):
    # build model
    model = SRLModel(config)
    model.build()
    model.restore_session(config.dir_model)
    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter, config.task)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter, config.task)

    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter, config.task)
    # train model
    params = model.train(train, dev, test)
    results = model.evaluate(test)
    finalResults.append(results)
    allParams.append(params)
    model.close_session()

def train_loop(config,objective):
    for layer in range(2,5):
        for step in [3]:
            for train_embeddings in [True]:
                for nepochs in [20, 100]:
                    for dropout in [0.5]:
                        for batch_size in [2, 32]:
                            for lr in [0.001]:
                                for lr_decay in [0.97]:
                                    for model_type in ['slstm']:
                                        config.layer = layer
                                        config.step = step
                                        config.train_embeddings = train_embeddings
                                        config.nepochs = nepochs
                                        config.dropout = dropout
                                        config.batch_size = batch_size
                                        config.lr = lr
                                        config.lr_decay = lr_decay
                                        config.model_type = model_type
                                        if objective == "train":
                                            config.model_name = model_type+"_lr"+str(lr)+"_batch"+str(batch_size)+"_layer"+str(layer)
                                            model_build_train_eval(config)
                                        elif objective == "retrain":
                                            model_build_retrain_eval(config)

## Train again the best model
def reTrain(config,model_name):
    config.model_name = model_name
    train_loop(config,"retrain")



def printResult(config):
    config.logger.info(finalResults)
    maxAcc = max([[res["acc"], finalResults.index(res)] for res in finalResults])
    paramsOfMaxAcc = allParams[maxAcc[1]]
    config.logger.info(allParams)
    config.logger.info("max acc: {:}".format(maxAcc[0]))
    config.logger.info("params of max acc: {:}".format(paramsOfMaxAcc))

def main():
    task = str(sys.argv[1]) # The type of model to train
    objective = str(sys.argv[2]) # objective of the run train or reTrain (Continue training the best model)

    build_data.main(task)
    if task == "predId":
        if objective == "train":
            # create instance of config
            config = PredIdConfig()
            train_loop(config,"train")
            printResult(config)
        elif objective == "retrain":
            model_name = str(sys.argv[3])
            if model_name:
                config = PredIdConfig()
                reTrain(config,model_name)
                printResult(config)
            else:
                print("Invalid command! 1")
        else:
            print("Invalid command! 2")

    elif task == "srlId":
        if objective == "train":
            # create instance of config
            config = SrlIdConfig()
            train_loop(config,"train")
            printResult(config)
        elif objective == "retrain":
            model_name = str(sys.argv[3])
            if model_name:
                config = SrlIdConfig()
                reTrain(config,model_name)
                printResult(config)
            else:
                print("Invalid command! 3")
        else:
            print("Invalid command! 4" )
    else:
        print("Invalid command! 5")



if __name__ == "__main__":
    main()
