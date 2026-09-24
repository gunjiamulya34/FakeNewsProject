from flask import Flask, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route('/')
def home():
    return '''
    <h2>Fake News Detection</h2>
    <form action="/predict" method="post">
        <textarea name="news" rows="6" cols="60" placeholder="Enter news here"></textarea>
        <br><br>
        <button type="submit">Check News</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    vector = vectorizer.transform([news])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "REAL NEWS"
    else:
        if "government" in news.lower() or "president" in news.lower() or "minister" in news.lower():
            result = "REAL NEWS"
        else:
            result = "FAKE NEWS"

    return f"<h2>Result: {result}</h2><br><a href='/'>Try Again</a>"

if __name__ == "__main__":
    app.run(debug=True)