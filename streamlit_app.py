import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ NLTK Resource Downloads (for Streamlit Cloud)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Load Saved Models & Data
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))
df = pd.read_csv("cleaned_tmdb_movies.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ✅ Text Cleaning Function
def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

# ✅ Streamlit Interface
st.title("🎬 Movie Genre Classifier")

option = st.radio("Choose input type:", ["Plot", "Movie Title"])

if option == "Plot":
    plot = st.text_area("Enter movie plot:")
    if st.button("Predict Genre"):
        cleaned = clean_text(plot)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        labels = mlb.inverse_transform(pred)
        st.write("🎯 Predicted Genres:", ", ".join(labels[0]) if labels else "No genre detected.")

else:
    title = st.text_input("Enter movie title:")
    if st.button("Predict Genre"):
        result = df[df["title"].str.lower() == title.lower()]
        if not result.empty:
            vec = vectorizer.transform(result["clean_overview"])
            pred = model.predict(vec)
            labels = mlb.inverse_transform(pred)
            st.write("🎯 Predicted Genres:", ", ".join(labels[0]) if labels else "No genre detected.")
        else:
            st.warning("Movie not found in dataset.")
