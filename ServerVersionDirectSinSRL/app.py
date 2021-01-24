from flask import Flask, request
from sinsrl.finalPredict import predictor
import flask_cors
import tensorflow as tf
import json

app = Flask(__name__, template_folder='.sinsrl/')
predictorIns = predictor(tf, "slstm_lr0.05_batch32_layer3", "slstm_lr0.05_batch32_layer2")


@app.route('/', methods=['GET'])
def main():
    return 'Request Success'


@app.route('/predict', methods=['POST'])
@flask_cors.cross_origin()
def predict():
    query_request = request.json
    print(query_request)
    results = predictorIns.predict(query_request["sinSentence"])

    result = {}
    result["result"] = json.dumps(results)
    return result


if __name__ == '__main__':
    app.run(debug=True)
