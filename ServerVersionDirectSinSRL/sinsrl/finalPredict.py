import sys
# sys.path.append('.')
from .model.srlIdConfig import SrlIdConfig
from .model.predIdConfig import PredIdConfig
from .model.srl_model import SRLModel
import json

class predictor:

    def findInsideTags(self, i, n, newTags, tags,
                       seqList):  # (i, 4, ['O', 'O', 'B-go.02', 'O'], ['O', 'O', 'B-go.02', 'O'], [])
        if (i == n - 1 or 'I-' not in tags[i + 1]):
            seqList.append(newTags)
        elif ('I-' in tags[i + 1]):
            newTags[i + 1] = tags[i + 1]
            self.findInsideTags(i + 1, n, newTags, tags, seqList)

    def predWiseTags(self, tokens, tags, seqList):  # (['මම', 'ගෙදර', 'යමි', '.'], ['O', 'O', 'B-go.02', 'O'], [])
        n = len(tokens)
        for i in range(n):
            if ('B-' in tags[i]):
                newTags = ['O' for i in range(n)]
                newTags[i] = tags[i]
                self.findInsideTags(i, n, newTags, tags, seqList)

    def makeInputToNext(self, words_raw, seq):  # (['මම', 'ගෙදර', 'යමි', '.'], ['O', 'O', 'B-go.02', 'O'])
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
        preds = self.predIdModel.predict(words_raw)  # ['O', 'O', 'B-go.02', 'O']
        seqList = []
        self.predWiseTags(words_raw, preds, seqList)

        # Predict SRL tags
        results = []
        for seq in seqList:  # [['O', 'O', 'B-go.02', 'O']]
            inputToNextModel = self.makeInputToNext(words_raw, seq)  # ['මම', 'ගෙදර', 'යමි-B-go.02', '.']
            finalPreds = self.srlIdModel.predict(inputToNextModel)  # ['B-ARG0', 'B-ARG1','pred' , 'O']
            output: list = self.displayOutput(inputToNextModel, finalPreds)  # ['B-ARG0', 'B-ARG1','B-go.02' , 'O']
            results.append(output)
        print (results)
        return self.processOutputAsSinSRL(words_raw, results)

# if __name__ == "__main__":
#     main()
