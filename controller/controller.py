from flask import Flask, request
from dao import dao


app = Flask(__name__)


def get_flask_app():
    return app


@app.route('/prediction', methods=("POST",))
def model_prediction():
    config = request.json
    print("input config : ", config)
    result = dao.main(config.get("FilePath"))
    return result
