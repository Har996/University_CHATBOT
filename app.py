from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
import string
from difflib import get_close_matches

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_excel('chatbot.xlsx')
data = data.dropna(subset=['Question', 'Answer'])

stop_words = set(nltk_stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(tokens)

questions = data['Question'].apply(preprocess_text).tolist()
answers = data['Answer'].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_response():
    user_msg = request.json['msg']
    processed = preprocess_text(user_msg)
    match = get_close_matches(processed, questions, n=1, cutoff=0.4)
    if match:
        idx = questions.index(match[0])
        return jsonify({'reply': answers[idx]})
    else:
        return jsonify({'reply': "Sorry, I couldn't find an answer for that."})

if __name__ == '__main__':
    app.run(debug=True)
