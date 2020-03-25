from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


@app.route('/')
def home():
    return "Hello flask app"


class predictApi(Resource):
    def post(self):
        return {'hello': 'flaskRestful'}


api.add_resource(predictApi, '/predictapi')

if __name__ == '__main__':
    app.run()
