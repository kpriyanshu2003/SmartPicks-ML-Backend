from flask import Flask, request, jsonify
from markupsafe import escape
import time
import json

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is working perfectly"})

@app.route("/api/<string:filename>", methods=["GET"])
def api(filename):
    data = {
        "text1": "hello world",
        "count": [i for i in range(10)]
    }
    json_data = json.dumps(data, indent=2)

    result = {"message": "Received data successfully", "data": json.loads(json_data)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False, passthrough_errors=True, use_debugger=False, use_reloader=True)
