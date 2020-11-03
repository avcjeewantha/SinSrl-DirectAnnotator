from model.srlIdConfig import SrlIdConfig
from model.predIdConfig import PredIdConfig
from model.srl_model import SRLModel


def findInsideTags(i, n, newTags, tags, seqList):
    if (i == n - 1):
        return newTags
    if ('I-' not in tags[i + 1]):
        seqList.append(newTags)
    else:
        newTags[i + 1] = tags[i + 1]
        if (i + 1 == n - 1):
            return newTags
        else:
            findInsideTags(i + 1, n, newTags)


def predWiseTags(tokens, tags, seqList):
    n = len(tokens)
    for i in range(n):
        if ('B-' in tags[i]):
            newTags = ['O' for i in range(n)]
            newTags[i] = tags[i]
            pred = tags[i].split('-')[-1]
            findInsideTags(i, n, newTags, tags, seqList)

def makeInputToNext(words_raw, seq):
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

def main():
    # create instance of config
    predConfig = PredIdConfig()
    # build model
    predModel = SRLModel(predConfig)
    predModel.build()
    predModel.restore_session(predConfig.dir_model)
    sentence = input("input> ")
    words_raw = sentence.strip().split(" ")
    if (words_raw[-1][-1] == '.' and words_raw[-1] != '.'):
        words_raw = words_raw[:-1] + [words_raw[-1][:-1]] + ['.']
    elif (words_raw[-1] != '.'):
        words_raw.append('.')
    preds = predModel.predict(words_raw)
    predModel.close_session()
    seqList = []
    predWiseTags(words_raw, preds, seqList)

    # create instance of config
    srlConfig = SrlIdConfig()
    # build model
    srlModel = SRLModel(srlConfig)
    srlModel.build()
    srlModel.restore_session(srlConfig.dir_model)
    for seq in seqList:
        inputToNextModel = makeInputToNext(words_raw, seq)
        finalPreds = srlModel.predict(inputToNextModel)
        print(words_raw, finalPreds)
    srlModel.close_session()

if __name__ == "__main__":
    main()
