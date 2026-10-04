import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABEL2ID = {"sale_oneoff": 0, "sale_enterprise": 1, "it_support": 2, "admin_general": 3}

# --- reproduce Track A (classical ML) on the same split ---
df = pd.read_csv("data/mock_conversations_400.csv")
df["tokens"] = df["text"].apply(lambda x: " ".join(word_tokenize(x, engine="newmm")))

X_train, X_test, y_train, y_test = train_test_split(
    df["tokens"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train_tfidf, y_train)
svm_preds = svm.predict(X_test_tfidf)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)
rf_preds = rf.predict(X_test_tfidf)

model_predictions = {
    "classical_svm": svm_preds,
    "classical_rf": rf_preds,
}

# --- load Track B (WangchanBERTa) predictions if available ---
bert_cache = Path("wangchanberta/preds_cache.pkl")
if bert_cache.exists():
    with open(bert_cache, "rb") as f:
        cached = pickle.load(f)
    assert list(cached["y_test"]) == list(y_test), "BERT predictions used a different split"
    model_predictions["wangchanberta"] = cached["bert_preds"]
else:
    print("[warn] wangchanberta/preds_cache.pkl not found — skipping wangchanberta")

# --- load Track C (LLM prompting) predictions if available ---
llm_cache = Path("llm_prompting/preds_cache.pkl")
if llm_cache.exists():
    with open(llm_cache, "rb") as f:
        cached = pickle.load(f)
    assert list(cached["y_test"]) == list(y_test), "LLM predictions used a different split"
    model_predictions["llm_prompting"] = cached["llm_preds"]
else:
    print("[warn] llm_prompting/preds_cache.pkl not found — skipping llm_prompting")

results = []
for model_name, preds in model_predictions.items():
    results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
    })

results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
print(results_df)
results_df.to_csv("evaluation/model_comparison_results.csv", index=False)

for model_name, preds in model_predictions.items():
    cm = confusion_matrix(y_test, preds, labels=list(LABEL2ID.keys()))
    print(f"\n{model_name} confusion matrix:\n{cm}")

with open("evaluation/preds_cache.pkl", "wb") as f:
    pickle.dump({"y_test": list(y_test), **model_predictions}, f)
