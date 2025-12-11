# app.py
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.joblib")
VECT_PATH = os.path.join("model", "vectorizer.joblib")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

label_map = {0: "Not Spam (Ham)", 1: "Spam"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            X = vectorizer.transform([message])
            pred = model.predict(X)[0]
            try:
                proba = model.predict_proba(X)[0].max()
            except Exception:
                proba = None
            prediction = label_map.get(int(pred), str(pred))
        else:
            prediction = "Please enter a message."
    return render_template("index.html", prediction=prediction, proba=proba, message=message)

if __name__ == "__main__":
    app.run(debug=True)
