# 🎬 Movie Genre Classification App

This is a machine learning web app that classifies a movie’s genre from either its **plot** or **title**, built using:
- `Streamlit` for the UI  
- `scikit-learn` for model training  
- `NLTK` for text preprocessing  

---

## 📦 Features

- ✅ Input movie plot ➜ get genre  
- ✅ Input movie title ➜ fetch plot and predict genre  
- ✅ Trained on real-world movie dataset  
- ✅ Runs via Colab + Streamlit + ngrok  

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/Arjo216/movie-genre-classifier.git
cd movie-genre-classifier
pip install -r requirements.txt
streamlit run streamlit_app.py
