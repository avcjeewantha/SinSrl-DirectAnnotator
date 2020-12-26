from model.data_utils import CoNLLDataset
from model.srlIdConfig import SrlIdConfig
from model.srl_model import SRLModel
from model.predIdConfig import PredIdConfig
import build_data
import sys
import os
import json

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
    params = model.train(train, dev, test, 0)
    results = model.evaluate(test)
    finalResults.append(results)
    allParams.append(params)
    model.close_session()


# For the re training process
def model_build_retrain_eval(config, best_score):
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
    params = model.train(train, dev, test, 0)
    results = model.evaluate(test)
    finalResults.append(results)
    allParams.append(params)
    model.close_session()


# process params with updated values since the model is already half run
def processParamsForRetrain(bestParams, params):
    newParams = {}
    for key, val in params.items():
        newParams[key] = val[val.index(bestParams[key]):]

    return newParams


def retrain_loop(config, params):
    with open(config.dir_model + 'bestModelParams.txt') as json_file:
        bestParams = json.load(json_file)
    json_file.close()

    newParams = processParamsForRetrain(bestParams, params)

    for layer in newParams["no of layers"]:
        for step in newParams["no of steps"]:
            for train_embeddings in newParams["train_embeddings"]:
                for nepochs in newParams["no of nepochs"]:
                    for dropout in newParams["dropout"]:
                        for batch_size in newParams["batch_size"]:
                            for lr in newParams["learning rate"]:
                                for lr_decay in newParams["lr_decay"]:
                                    for model_type in newParams["model_type"]:
                                        config.layer = layer
                                        config.step = step
                                        config.train_embeddings = train_embeddings
                                        config.nepochs = nepochs
                                        config.dropout = dropout
                                        config.batch_size = batch_size
                                        config.lr = lr
                                        config.lr_decay = lr_decay
                                        config.model_type = model_type
                                        model_build_retrain_eval(config, bestParams["best_score"])


def train_loop(config, params):
    for layer in params["no of layers"]:
        for step in params["no of steps"]:
            for train_embeddings in params["train_embeddings"]:
                for nepochs in params["no of nepochs"]:
                    for dropout in params["dropout"]:
                        for batch_size in params["batch_size"]:
                            for lr in params["learning rate"]:
                                for lr_decay in params["lr_decay"]:
                                    for model_type in params["model_type"]:
                                        config.layer = layer
                                        config.step = step
                                        config.train_embeddings = train_embeddings
                                        config.nepochs = nepochs
                                        config.dropout = dropout
                                        config.batch_size = batch_size
                                        config.lr = lr
                                        config.lr_decay = lr_decay
                                        config.model_type = model_type

                                        model_name = model_type + "_lr" + str(lr) + "_batch" + str(
                                            batch_size) + "_layer" + str(layer) + "/"
                                        config.dir_model = config.dir_model_root + model_name
                                        model_build_train_eval(config)


## Train again the defined model
def reTrain(config, params, model_name):
    config.dir_model = config.dir_model_root + model_name + "/"
    retrain_loop(config, params)


def printResult(config):
    config.logger.info(finalResults)
    maxAcc = max([[res["acc"], finalResults.index(res)] for res in finalResults])
    paramsOfMaxAcc = allParams[maxAcc[1]]
    config.logger.info(allParams)
    config.logger.info("max acc: {:}".format(maxAcc[0]))
    config.logger.info("params of max acc: {:}".format(paramsOfMaxAcc))


def main():
    task = str(sys.argv[1])  # The type of model to train
    objective = str(sys.argv[2])  # objective of the run train or reTrain (Continue training the best model)

    build_data.main(task)

    with open('parameters.json') as json_file:
        params = json.load(json_file)
    json_file.close()

    if task == "predId":
        if objective == "train":
            # create instance of config
            config = PredIdConfig()
            train_loop(config, params)
            printResult(config)
        elif objective == "retrain":
            model_name = str(sys.argv[3])
            if model_name:
                config = PredIdConfig()
                reTrain(config, params, model_name)
                printResult(config)
            else:
                print("Invalid command! 1")
        else:
            print("Invalid command! 2")

    elif task == "srlId":
        if objective == "train":
            # create instance of config
            config = SrlIdConfig()
            train_loop(config, "train")
            printResult(config)
        elif objective == "retrain":
            model_name = str(sys.argv[3])
            if model_name:
                config = SrlIdConfig()
                reTrain(config, params, model_name)
                printResult(config)
            else:
                print("Invalid command! 3")
        else:
            print("Invalid command! 4")
    else:
        print("Invalid command! 5")


if __name__ == "__main__":
    main()
