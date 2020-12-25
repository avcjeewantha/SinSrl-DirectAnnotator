from model.srlIdConfig import SrlIdConfig
from model.predIdConfig import PredIdConfig
from model.srl_model import SRLModel


def findInsideTags(i, n, newTags, tags, seqList): #(i, 4, ['O', 'O', 'B-go.02', 'O'], ['O', 'O', 'B-go.02', 'O'], [])
    if (i == n - 1 or 'I-' not in tags[i + 1]):
        seqList.append(newTags)
    elif('I-' in tags[i + 1]):
        newTags[i + 1] = tags[i + 1]
        findInsideTags(i + 1, n, newTags, tags, seqList)


def predWiseTags(tokens, tags, seqList): #(['මම', 'ගෙදර', 'යමි', '.'], ['O', 'O', 'B-go.02', 'O'], [])
    n = len(tokens)
    for i in range(n):
        if ('B-' in tags[i]):
            newTags = ['O' for i in range(n)]
            newTags[i] = tags[i]
            findInsideTags(i, n, newTags, tags, seqList)

def makeInputToNext(words_raw, seq): # (['මම', 'ගෙදර', 'යමි', '.'], ['O', 'O', 'B-go.02', 'O'])
    n = len(seq)
    newInput = []
    for i in range(n):
        sinWord = words_raw[i]
        if(seq[i] != 'O'):
            tag = seq[i]
            newSinWord = sinWord+"-"+tag
            newInput.append(newSinWord)
        else:
            newInput.append(sinWord)
    return newInput

def displayOutput(words_raw, seq):
	for i in range(len(seq)):
		if('-' in words_raw[i] and '.0' in words_raw[i]):
			seq[i] = words_raw[i].split('-', 1)[-1]
	return seq

def main():

    model_name = str(sys.argv[1]) # The name of the model to use for prediction
    if model_name:
        # create instance of config
        predConfig = PredIdConfig()

        #update the model path
        predConfig.dir_model = predConfig.dir_model_root+model_name

        # build model
        predModel = SRLModel(predConfig)
        predModel.build()
        predModel.restore_session(predConfig.dir_model)
        sentence = input("input> ")   #මම ගෙදර යමි .
        words_raw = sentence.strip().split(" ") #['මම', 'ගෙදර', 'යමි', '.']
        if (words_raw[-1][-1] == '.' and words_raw[-1] != '.'):
            words_raw = words_raw[:-1] + [words_raw[-1][:-1]] + ['.']
        elif (words_raw[-1] != '.'):
            words_raw.append('.')
        preds = predModel.predict(words_raw)    #['O', 'O', 'B-go.02', 'O']
        predModel.close_session()
        seqList = []
        predWiseTags(words_raw, preds, seqList)

        # create instance of config
        srlConfig = SrlIdConfig()
        # build model
        srlModel = SRLModel(srlConfig)
        srlModel.build()
        srlModel.restore_session(srlConfig.dir_model)
        for seq in seqList: #[['O', 'O', 'B-go.02', 'O']]
            inputToNextModel = makeInputToNext(words_raw, seq) #['මම', 'ගෙදර', 'යමි-B-go.02', '.']
            finalPreds = srlModel.predict(inputToNextModel) #['B-ARG0', 'B-ARG1','pred' , 'O']
            output = displayOutput(inputToNextModel, finalPreds) #['B-ARG0', 'B-ARG1','B-go.02' , 'O']
            print(output)
        srlModel.close_session()

    else:
        print("Invalid command. PLease enter the name of the prediction model !")

if __name__ == "__main__":
    main()
