from flask import Flask, request  # import flask
from flask_restful import Resource, Api  # for API building
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json

app = Flask(__name__)  # create an app instance
api = Api(app)  # integrating restful api for app

stemmer = LancasterStemmer()

# data preprocessing for train DNN
with open("TrainedAlgorithms/DnnKnowledgeModel/intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

# Traning DNN

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try to open model or train new

model.load("TrainedAlgorithms/DnnKnowledgeModel/model.tflearn")


# Prediction

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(inp):
    print("Start talking with the bot (type quit to stop)!")

    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)





class DnnKnowledgeApi(Resource):
    def post(self):
        query_json = request.get_json(force=True)
        query = query_json['query']
        response = chat(query)
        return {"response" :response }, 201


api.add_resource(DnnKnowledgeApi, '/dnnapi')


class Home(Resource):
    def get(self):
        return "Use the API"


api.add_resource(Home, '/')

if __name__ == "__main__":  # on running python app.py
    app.run(debug=True)  # run with debug mood
