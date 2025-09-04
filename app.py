import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # lowercasing
    text = nltk.word_tokenize(text)  # word tokenization
    # print(text)

    y = []  # removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # y is cloned to text variable. variable cannot directly be assigned to a list.
    y.clear()  # clears y. Its again a new empty list

    for i in text:  # Removing stop words and punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:  # Stemming
        y.append(ps.stem(i))

    # print(y)
    # print(" ".join(y))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_input('Enter Email Message')

if st.button('Predict'):

    #1. Preprocess
    transformed_sms = transform_text(input_sms)

    #2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    #3. Predict
    result = model.predict(vector_input)

    #4. Display
    if result == 1:
        st.header("SPAM!")
    else:
        st.header("NOT SPAM")


# Streamlit v1.49.1
# nltk v3.9.1