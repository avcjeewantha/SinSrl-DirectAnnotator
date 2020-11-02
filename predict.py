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


def interactive_shell(model):
    """Creates interactive shell to play with model
    Args:
        model: instance of SRLModel
    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> මම ගෙදර යමි""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


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

    # interact
    interactive_shell(model)


if __name__ == "__main__":
    main()
