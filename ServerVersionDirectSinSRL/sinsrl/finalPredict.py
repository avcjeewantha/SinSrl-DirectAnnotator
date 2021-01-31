import sys
# sys.path.append('.')
from .model.srlIdConfig import SrlIdConfig
from .model.predIdConfig import PredIdConfig
from .model.srl_model import SRLModel
import json

class predictor:

    def predWiseTags(self, tags):  # ['O', 'O', 'go.02', 'O']
        seqList = []
        predicates = []
        last = ''
        for i in range(len(tags)):
            if (tags[i] != 'O') and (tags[i] != last):
                predicates.append(i)
                last = tags[i]
        for i in range(len(predicates)):
            seq = ['O'] * len(tags)
            if i != len(predicates) - 1:
                for j in range(predicates[i], predicates[i + 1]):
                    seq[j] = tags[j]
            else:
                for j in range(predicates[i], len(tags)):
                    seq[j] = tags[j]
            seqList.append(seq)
        return seqList

    def makeInputToNext(self, words_raw, seq):  # (['මම', 'ගෙදර', 'යමි', '.'], ['O', 'O', 'go.02', 'O'])
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

    def displayOutput(self, words_raw, seq):
        for i in range(len(seq)):
            if ('-' in words_raw[i] and '.' in words_raw[i]):
                seq[i] = words_raw[i].split('-', 1)[-1]
        seq = [x if x!="pred" else "O" for x in seq]
        return seq

    def __init__(self, tf, predIdModel, srlIdModel):

        # create instance of config
        predConfig = PredIdConfig()
        srlConfig = SrlIdConfig()

        # update the model path
        predConfig.dir_model = predConfig.dir_model_root + predIdModel + "/"
        srlConfig.dir_model = srlConfig.dir_model_root + srlIdModel + "/"

        # build models
        predId_graph = tf.Graph()
        with predId_graph.as_default():
            self.predIdModel = SRLModel(predConfig)

        srlId_graph = tf.Graph()
        with srlId_graph.as_default():
            self.srlIdModel = SRLModel(srlConfig)

        srl_sess = tf.Session(graph=srlId_graph)
        pred_sess = tf.Session(graph=predId_graph)

        try:
            with pred_sess.as_default():
                with predId_graph.as_default():
                    tf.global_variables_initializer().run()
                    self.predIdModel.build()
                    self.predIdModel.restore_session(predConfig.dir_model)

            with srl_sess.as_default():
                with srlId_graph.as_default():
                    tf.global_variables_initializer().run()
                    self.srlIdModel.build()
                    self.srlIdModel.restore_session(srlConfig.dir_model)

            # srl_sess.close()
            # pred_sess.close()
        except:
            print("Prediction models not found !")

    def processOutputAsSinSRL(self, words, results):
        tokenJsonLst = []
        for word in words:
            tokenJsonObj = {}
            tags = []
            for resultLst in results:
                tags.append(resultLst[words.index(word)])

            tokenJsonObj["text"] = word
            tokenJsonObj["frame"] = str(tags)
            tokenJsonLst.append(json.dumps(tokenJsonObj))
        return tokenJsonLst



    def predict(self, sentence):
        # Predict Predicates
        #     while 1:
        #     sentence = input("input> ")  # මම ගෙදර යමි .
        words_raw = sentence.strip().split(" ")  # ['මම', 'ගෙදර', 'යමි', '.']
        if (words_raw[-1][-1] == '.' and words_raw[-1] != '.'):
            words_raw = words_raw[:-1] + [words_raw[-1][:-1]] + ['.']
        elif (words_raw[-1] != '.'):
            words_raw.append('.')
        preds = self.predIdModel.predict(words_raw)  # ['O', 'O', 'go.02', 'O']
        seqList = self.predWiseTags(preds)

        # Predict SRL tags
        results = []
        for seq in seqList:  # [['O', 'O', 'go.02', 'O']]
            inputToNextModel = self.makeInputToNext(words_raw, seq)  # ['මම', 'ගෙදර', 'යමි-go.02', '.']
            finalPreds = self.srlIdModel.predict(inputToNextModel)  # ['ARG0', 'ARG1','pred' , 'O']
            output: list = self.displayOutput(inputToNextModel, finalPreds)  # ['ARG0', 'ARG1','go.02' , 'O']
            results.append(output)
        print (results)
        return self.processOutputAsSinSRL(words_raw, results)

# if __name__ == "__main__":
#     main()
