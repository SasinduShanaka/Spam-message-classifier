# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# --- load dataset ---
sms_path = os.path.join("dataset", "SMSSpamCollection")
if os.path.exists(sms_path):
    df = pd.read_csv(sms_path, sep='\t', header=None, names=['label','message'], encoding='utf-8')
else:
    csv_path = os.path.join("data", "spam.csv")
    df = pd.read_csv(csv_path, encoding='latin-1')
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1','v2']].rename(columns={'v1':'label','v2':'message'})

# quick cleaning
df['label'] = df['label'].astype(str).str.strip().str.lower()
df = df[df['message'].notnull()]

# convert label to binary
df['label_bin'] = df['label'].map({'ham':0, 'spam':1})
df = df.dropna(subset=['label_bin'])

X = df['message'].values
y = df['label_bin'].values

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

# vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# eval
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=['ham','spam']))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# save artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")
print("Saved model -> model/model.joblib")
print("Saved vectorizer -> model/vectorizer.joblib")
