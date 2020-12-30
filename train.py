from model.data_utils import CoNLLDataset
from model.srlIdConfig import SrlIdConfig
from model.srl_model import SRLModel
from model.predIdConfig import PredIdConfig
from pathlib import Path
import shutil
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
    if os.path.isfile(config.dir_model_root + 'modelResults.json') and os.access(
            config.dir_model_root + 'modelsResults.json',
            os.R_OK):

        params = model.train(train, dev, test, 0)
        results = model.evaluate(test)
        finalResults.append(results)
        allParams.append(params)

        with open(config.dir_model_root + 'modelResults.json') as modelResults:
            allModelStats = json.load(modelResults)

            allModelStats["finalResults"] = finalResults  # save model accuracy and parameter states for retrain
            allModelStats["allParams"] = allParams
        modelResults.close()

        with open(config.dir_model_root + 'modelResults.json', 'w') as modelResults:
            json.dump(allModelStats, modelResults)  # write model stats into file
            model.close_session()

        modelResults.close()
    else:
        params = model.train(train, dev, test, 0)
        results = model.evaluate(test)
        finalResults.append(results)
        allParams.append(params)
        with open(config.dir_model_root + 'modelResults.json', 'w') as json_file:
            model_stats = {"finalResults": finalResults, "allParams": allParams}
            json.dump(model_stats, json_file)  # write model stats into file

            model.close_session()
        json_file.close()


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

    with open(config.dir_model_root + 'modelResults.json') as json_file:
        allModelStats = json.load(json_file)
        finalResults = allModelStats["finalResults"]  # replace model stats from saved file
        allParams = allModelStats["allParams"]

        params = model.train(train, dev, test, best_score)
        results = model.evaluate(test)
        finalResults.append(results)  # append new stats into the list
        allParams.append(params)

        allModelStats["finalResults"] = finalResults  # save model accuracy and parameter states for retrain
        allModelStats["allParams"] = allParams
    json_file.close()

    with open(config.dir_model_root + 'modelResults.json', 'w') as json_file:
        json.dump(allModelStats, json_file)  # write model stats into file
        model.close_session()

    json_file.close()


def processParamsForRetrain(bestParams, params):
    """process params with updated values since the model is already half run"""
    paramCombinations = [] # all parameter combinations
    for layer in params["no of layers"]:
        for step in params["no of steps"]:
            for train_embeddings in params["train_embeddings"]:
                for nepochs in params["no of nepochs"]:
                    for dropout in params["dropout"]:
                        for batch_size in params["batch_size"]:
                            for lr in params["learning rate"]:
                                for lr_decay in params["lr_decay"]:
                                    for model_type in params["model_type"]:
                                        paramCombinations.append(
                                            [layer, step, train_embeddings, nepochs, dropout, batch_size, lr, lr_decay,
                                             model_type])

    bestParamsCombination = list(bestParams.values())[:-1] # last trained best parameter combination
    paramCombinationsToIterate = paramCombinations[paramCombinations.index(bestParamsCombination):] # parameter combinations more to run

    return paramCombinationsToIterate


def retrain_loop(config, params):
    with open(config.dir_model + 'bestModelParams.txt') as json_file:
        bestParams = json.load(json_file)
    json_file.close()

    newParams = processParamsForRetrain(bestParams, params)

    for paramset in newParams:
        config.layer = paramset[0]
        config.step = paramset[1]
        config.train_embeddings = paramset[2]
        config.nepochs = paramset[3]
        config.dropout = paramset[4]
        config.batch_size = paramset[5]
        config.lr = paramset[6]
        config.lr_decay = paramset[7]
        config.model_type = paramset[8]

        model_name = paramset[8] + "_lr" + str(paramset[6]) + "_batch" + str(
                                            paramset[5]) + "_layer" + str(paramset[0]) + "/"
        config.dir_model = config.dir_model_root + model_name


        if newParams.index(paramset) != 0:
            model_build_train_eval(config)
        else:
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
    accList = [[res["f1"], finalResults.index(res)] for res in finalResults]
    maxAcc = max(accList, key=lambda x: x[0])
    paramsOfMaxAcc = allParams[maxAcc[1]]
    config.logger.info(allParams)
    config.logger.info("max acc: {:}".format(maxAcc[0]))
    config.logger.info("params of max acc: {:}".format(paramsOfMaxAcc))


def main():
    task = str(sys.argv[1])  # The type of model to train
    objective = str(sys.argv[2])  # objective of the run train or reTrain (Continue training the best model)

    with open('parameters.json') as json_file:
        params = json.load(json_file)
    json_file.close()


    if task == "predId":
        if objective == "train":

            # Remove previous results
            dirpath = Path('results/')
            if  dirpath.exists() and dirpath.is_dir():
                shutil.rmtree(dirpath)

            build_data.main(task)

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
            # Remove previous results
            dirpath = Path('results/')
            if  dirpath.exists() and dirpath.is_dir():
                shutil.rmtree(dirpath)

            build_data.main(task)

            # create instance of config
            config = SrlIdConfig()
            train_loop(config, params)
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
