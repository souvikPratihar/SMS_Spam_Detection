import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Download stopwords only (punkt is NOT needed)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return vectorizer, model

tfidf, model = load_model_and_vectorizer()

st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

def transform_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)  # ✅ safe tokenization without punkt

    filtered = []
    for word in tokens:
        if word not in stopwords.words('english') and word not in string.punctuation:
            filtered.append(ps.stem(word))

    return ' '.join(filtered)

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("🚨 SPAM 😤")
    else:
        st.header("✅ NOT SPAM 😄")
