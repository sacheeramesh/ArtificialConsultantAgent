from flask import Flask, render_template
from flask import request

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello flask app"


@app.route('/predictapi', methods=['POST'])
def predict():
    return "api"


if __name__ == '__main__':
    app.run()
