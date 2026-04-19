import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

print("Loading dataset...")
fake = pd.read_csv("../data/Fake.csv")
real = pd.read_csv("../data/True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df.dropna(subset=["content"], inplace=True)

print(f"Total samples: {len(df)}")
print(f"Fake: {(df.label==0).sum()} | Real: {(df.label==1).sum()}")

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("clf", PassiveAggressiveClassifier(max_iter=50, random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

joblib.dump(pipeline, "model.pkl")
print("Model saved to model/model.pkl")