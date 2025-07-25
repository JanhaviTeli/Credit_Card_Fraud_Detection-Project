from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route("/check_merchant", methods=["POST"])
def check_merchant():
    data = request.get_json()
    merchant_id = data.get("merchant_id")
    if merchant_id == "M00005":
        return jsonify({"valid": True})
    else:
        return jsonify({"valid": False})

if __name__ == "__main__":
    app.run(debug=True)
