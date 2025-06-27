
import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model files (uploaded to Colab)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))

# Preprocess function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# Streamlit App UI
st.set_page_config(page_title="🎬 Movie Genre Classifier", layout="centered")
st.title("🎬 Movie Genre Classifier")
st.write("Enter a movie plot and I'll predict the genres.")

plot = st.text_area("Enter Movie Plot", height=200)

if st.button("Predict Genre"):
    if not plot.strip():
        st.warning("⚠️ Please enter a movie plot.")
    else:
        cleaned = clean_text(plot)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)
        genres = mlb.inverse_transform(pred)
        st.success(f"🎯 Predicted Genre(s): {', '.join(genres[0]) if genres and genres[0] else 'None'}")
