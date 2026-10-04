import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/mock_conversations_400.csv")
df["tokens"] = df["text"].apply(lambda x: " ".join(word_tokenize(x, engine="newmm")))

X_train, X_test, y_train, y_test = train_test_split(
    df["tokens"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
