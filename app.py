from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# Load the translation model and tokenizers
translation_model = pickle.load(open('simple_rnn_model.pkl', 'rb'))
english_tokenizer = pickle.load(open('english_tokenizer.pkl', 'rb'))
french_tokenizer = pickle.load(open('french_tokenizer.pkl', 'rb'))

# Define the logits_to_text function
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def translate_text(text):
    sentence = [english_tokenizer.word_index.get(word, 0) for word in text.split()]
    sentence = pad_sequences([sentence], maxlen=translation_model.input_shape[1], padding='post')
    translation = logits_to_text(translation_model.predict(sentence[:1])[0], french_tokenizer)
    return translation

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
            text = request.form.get('text', '')
            if text:
                translated_text = translate_text(text)
                return render_template('index.html', translated_text=translated_text, error='')
            else:
                return render_template('index.html', translated_text='', error='Text parameter missing')
    return render_template('index.html', translated_text='', error='')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)