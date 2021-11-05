from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from extract import *

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route("/extract", methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def extract():
    # try:
    opt, text = getData(request.json)
    print(opt, text)
    return extractEntity(opt,[text])
    # # except:
    # #     return jsonify({'state': 'ko'})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
