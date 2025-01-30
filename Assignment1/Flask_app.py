from flask import Flask, request, jsonify
import pickle
import numpy as np
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
app = Flask(__name__)
@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API! Use /predict to classify reviews."
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review_text = data.get("review_text", "")

        if not review_text:
            return jsonify({"error": "No review text provided"}), 400

        review_vector = vectorizer.transform([review_text])

        prediction = model.predict(review_vector)[0]
        sentiment = "positive" if prediction == 1 else "negative"

        return jsonify({"sentiment_prediction": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
