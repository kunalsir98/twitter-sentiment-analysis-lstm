import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import string

# Load the trained model and tokenizer
model = load_model('model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load stopwords and stemmer
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Define the Streamlit app
def main():
    st.title("Sentiment analysis app")
    st.write("Enter a message to check if it's positive or negative.")

    user_input = st.text_area("Enter your message here:")

    if st.button("Predict"):
        if user_input:
            cleaned_text = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(seq, maxlen=300)
            pred = model.predict(padded)

            if pred < 0.9:
                st.write("The message is **not positive**.")
            else:
                st.write("The message is **positive**.")
        else:
            st.write("Please enter a message.")

if __name__ == '__main__':
    main()