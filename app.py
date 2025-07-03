import streamlit as st
import pickle
import pandas as pd
import os
from datetime import datetime
from preprocessing import clean_text
os.makedirs("app", exist_ok=True)


# === Load model and vectorizer ===
model = pickle.load(open('models/sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# === App title ===
st.title("🎬 Movie Review Sentiment Analyzer")

# === User input ===
review = st.text_area("Enter a movie review below:")

# === Function to save prediction to history.csv ===
def save_history(review_text, prediction_label):
    history_file = 'app/history.csv'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create DataFrame for current entry
    new_entry = pd.DataFrame([{
        'Timestamp': timestamp,
        'Review': review_text,
        'Prediction': prediction_label
    }])

    # Append or create file
    if os.path.exists(history_file):
        new_entry.to_csv(history_file, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(history_file, index=False)

# === Predict Sentiment ===
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        sentiment = "👍 Positive" if prediction == 1 else "👎 Negative"
        st.subheader("Sentiment Prediction:")
        st.write(sentiment)

        # Save to history
        save_history(review, sentiment)

# === Display Prediction History ===
st.subheader("📜 Prediction History")

history_path = 'app/history.csv'

if os.path.exists(history_path):
    history_df = pd.read_csv(history_path)

    # Show only last 10 records
    st.dataframe(history_df.tail(10))

    # === Download Button ===
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download History CSV", data=csv, file_name="prediction_history.csv", mime='text/csv')

    # === Clear History Button ===
    if st.button("🗑️ Clear History"):
        os.remove(history_path)
        st.success("History cleared successfully!")
        st.rerun()

else:
    st.info("No history found yet. Make a prediction to see history here.")