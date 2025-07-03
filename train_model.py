import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk import download
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download NLTK stopwords
download('stopwords')
stop_words = set(stopwords.words('english'))

# === Preprocessing Function ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)            # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)         # Remove special characters and digits
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize whitespace
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# === Load Data ===
df = pd.read_csv(r"C:\Users\sakti\Downloads\IMDB Dataset.csv")

# Map sentiment to binary labels
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Clean text
df['cleaned'] = df['review'].apply(clean_text)

# === Vectorization ===
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment']

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Training ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === Save Model and Vectorizer ===
pickle.dump(model, open('models/sentiment_model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

print("\n✅ Model and vectorizer saved successfully!")
