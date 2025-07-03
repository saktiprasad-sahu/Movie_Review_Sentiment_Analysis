---
title: Movie Review Sentiment Analyzer
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: "1.34.0"
app_file: app.py
pinned: false
---

# 🎬 Movie Review Sentiment Analyzer

This app lets you input a movie review and predicts whether the sentiment is *Positive* or *Negative* using a machine learning model trained on the IMDB dataset.

✅ Features:
- Text preprocessing
- TF-IDF vectorizer
- Logistic Regression classifier
- Streamlit UI
- History logging
- Downlode and Clear History

---

📍 Built with Python, Scikit-learn, Pandas, NLTK, and Streamlit.

🧠 Trained on the [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

---

## 🚀 Run the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
