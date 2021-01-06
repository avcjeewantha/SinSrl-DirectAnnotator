from model.srlIdConfig import SrlIdConfig
from model.predIdConfig import PredIdConfig
from model.srl_model import SRLModel
import tensorflow as tf
import sys

def findInsideTags(i, n, newTags, tags, seqList):  # (i, 4, ['O', 'O', 'B-go.02', 'O'], ['O', 'O', 'B-go.02', 'O'], [])
    if (i == n - 1 or 'I-' not in tags[i + 1]):
        seqList.append(newTags)
    elif ('I-' in tags[i + 1]):
        newTags[i + 1] = tags[i + 1]
        findInsideTags(i + 1, n, newTags, tags, seqList)


def predWiseTags(tokens, tags, seqList):  # (['මම', 'ගෙදර', 'යමි', '.'], ['O', 'O', 'B-go.02', 'O'], [])
    n = len(tokens)
    for i in range(n):
        if ('B-' in tags[i]):
            newTags = ['O' for i in range(n)]
            newTags[i] = tags[i]
            findInsideTags(i, n, newTags, tags, seqList)


def makeInputToNext(words_raw, seq):  # (['මම', 'ගෙදර', 'යමි', '.'], ['O', 'O', 'B-go.02', 'O'])
    n = len(seq)
    newInput = []
    for i in range(n):
        sinWord = words_raw[i]
        if (seq[i] != 'O'):
            tag = seq[i]
            newSinWord = sinWord + "-" + tag
            newInput.append(newSinWord)
        else:
            newInput.append(sinWord)
    return newInput


def displayOutput(words_raw, seq):
    for i in range(len(seq)):
        if ('-' in words_raw[i] and '.' in words_raw[i]):
            seq[i] = words_raw[i].split('-', 1)[-1]
    seq = [x if x!="pred" else "O" for x in seq]
    return seq


def main():
    predId_model_name = ''
    srlId_model_name = ''
    predId_model_name = str(sys.argv[1])  # The name of the model to use for prediction
    srlId_model_name = str(sys.argv[2])

    if predId_model_name != '' and srlId_model_name != '':
        # create instance of config
        predConfig = PredIdConfig()
        srlConfig = SrlIdConfig()

        # update the model path
        predConfig.dir_model = predConfig.dir_model_root + predId_model_name + "/"
        srlConfig.dir_model = srlConfig.dir_model_root + srlId_model_name + "/"

        # build models
        predId_graph = tf.Graph()
        with predId_graph.as_default():
            predIdModel = SRLModel(predConfig)

        srlId_graph = tf.Graph()
        with srlId_graph.as_default():
            srlIdModel = SRLModel(srlConfig)

        srl_sess = tf.Session(graph=srlId_graph)
        pred_sess = tf.Session(graph=predId_graph)

        with pred_sess.as_default():
            with predId_graph.as_default():
                tf.global_variables_initializer().run()
                predIdModel.build()
                predIdModel.restore_session(predConfig.dir_model)

        with srl_sess.as_default():
            with srlId_graph.as_default():
                tf.global_variables_initializer().run()
                srlIdModel.build()
                srlIdModel.restore_session(srlConfig.dir_model)

        # Predict Predicates
        while 1:
            sentence = input("input> ")  # මම ගෙදර යමි .
            words_raw = sentence.strip().split(" ")  # ['මම', 'ගෙදර', 'යමි', '.']
            if (words_raw[-1][-1] == '.' and words_raw[-1] != '.'):
                words_raw = words_raw[:-1] + [words_raw[-1][:-1]] + ['.']
            elif (words_raw[-1] != '.'):
                words_raw.append('.')
            preds = predIdModel.predict(words_raw)  # ['O', 'O', 'B-go.02', 'O']
            seqList = []
            predWiseTags(words_raw, preds, seqList)

            # Predict SRL tags
            for seq in seqList:  # [['O', 'O', 'B-go.02', 'O']]
                inputToNextModel = makeInputToNext(words_raw, seq)  # ['මම', 'ගෙදර', 'යමි-B-go.02', '.']
                finalPreds = srlIdModel.predict(inputToNextModel)  # ['B-ARG0', 'B-ARG1','pred' , 'O']
                output = displayOutput(inputToNextModel, finalPreds)  # ['B-ARG0', 'B-ARG1','B-go.02' , 'O']
                print(output)

        # srl_sess.close()
        # pred_sess.close()

    else:
        print("Invalid command. PLease enter the name of the prediction model !")


if __name__ == "__main__":
    main()
