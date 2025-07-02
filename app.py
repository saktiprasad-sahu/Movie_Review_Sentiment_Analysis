import streamlit as st
import pandas as pd
import numpy as np
import nltk
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from AttentionLayer import Attention

# Ensure folders exist
os.makedirs("model", exist_ok=True)
os.makedirs("tokenizer", exist_ok=True)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Constants
MAX_VOCAB = 10000
MAX_LEN = 200
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
DATASET_PATH = "IMDB Dataset.csv"

# Preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Train model if not exists
def train_model():
    df = pd.read_csv(DATASET_PATH)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df['cleaned'] = df['review'].apply(preprocess_text)

    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    tokenizer.fit_on_texts(df['cleaned'])
    X = pad_sequences(tokenizer.texts_to_sequences(df['cleaned']), maxlen=MAX_LEN)
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    input_layer = Input(shape=(MAX_LEN,))
    embedding = Embedding(input_dim=MAX_VOCAB, output_dim=128)(input_layer)
    lstm = LSTM(64, return_sequences=True)(embedding)
    attention = Attention()(lstm)
    dropout = Dropout(0.5)(attention)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

    early_stop = EarlyStopping(patience=2, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1,
              class_weight=class_weights_dict, callbacks=[early_stop])

    # Save model & tokenizer
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    st.success("✅ Model trained and saved successfully!")

# Load model/tokenizer
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        st.warning("Training model as it does not exist...")
        train_model()
    model = load_model(MODEL_PATH, custom_objects={'Attention': Attention})
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0][0]

    # Confidence-based classification
    if pred >= 0.58:
        label = "💚 Positive"
    else:
        label = "💔 Negative"

    return label, pred


# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analyzer")

model, tokenizer = load_resources()

user_input = st.text_area("Enter your movie review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        label, score = predict_sentiment(user_input, model, tokenizer)
        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"**Confidence Score:** `{score:.2f}`")

st.markdown("---")
st.caption("Trained using IMDB dataset with custom Attention-based LSTM model.")
