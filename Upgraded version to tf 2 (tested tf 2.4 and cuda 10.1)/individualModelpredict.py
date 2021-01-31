from model.srlIdConfig import SrlIdConfig
from model.srl_model import SRLModel
from model.predIdConfig import PredIdConfig
import sys


def align_data(data):
    """Given dict with lists, creates aligned strings
    Adapted from Assignment 3 of CS224N
    Args:
        data: (dict) data["x"] = ["මම", "ගෙදර", "යමි"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "මම ගෙදර යමි"
                           data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def interactive_shell(model, task):
    """Creates interactive shell to play with model
    Args:
        model: instance of SRLModel
        task: predId or srlId
    """
    if task == "predId":
        model.logger.info("""This is an interactive mode. To exit, enter 'exit'. You can enter a sentence like , 
        input> මම ගෙදර යමි .""")
    elif task == "srlId":
        model.logger.info("""This is an interactive mode. To exit, enter 'exit'. You can enter a sentence like , 
            input> මම ගෙදර යමි-B-go.02 .""")
    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
            # sentence = "මම ගෙදර යමි ."
        except NameError:
            # for python 3
            sentence = input("input> ")
        words_raw = sentence.strip().split(" ")
        if words_raw[-1][-1] == '.' and words_raw[-1] != '.':
            words_raw = words_raw[:-1] + [words_raw[-1][:-1]] + ['.']
        elif words_raw[-1] != '.':
            words_raw.append('.')

        if words_raw == ["exit"]:
            break
        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main():
    global config
    task = str(sys.argv[1])
    model_name = str(sys.argv[2])  # The name of the model to use for prediction
    if model_name:
        if task == "predId":
            # create instance of config
            config = PredIdConfig()
            config.dir_model = config.dir_model_root + model_name + "/"
        elif task == "srlId":
            # create instance of config
            config = SrlIdConfig()
            config.dir_model = config.dir_model_root + model_name + "/"

        # build model
        model = SRLModel(config)
        model.build()
        model.restore_session(config.dir_model)
        # interact
        interactive_shell(model, task)
    else:
        print("Enter the model name to predict...")


if __name__ == "__main__":
    main()
