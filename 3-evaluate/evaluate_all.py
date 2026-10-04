import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize

SCRIPT_DIR = Path(__file__).resolve().parent              # evaluate/
ROOT = SCRIPT_DIR.parent                                  # repo root
RESULTS_DIR = SCRIPT_DIR / "results"                      # evaluate/results/ (output only, separate from code)
RESULTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "2-train-test-model"))          # so wangchanberta/llm_prompting are importable

LABEL2ID = {"sale_oneoff": 0, "sale_enterprise": 1, "it_support": 2, "admin_general": 3}

# --- reproduce Track A (classical ML) on the same split ---
df = pd.read_csv(ROOT / "1-data" / "mock_conversations_400.csv")
df["tokens"] = df["text"].apply(lambda x: " ".join(word_tokenize(x, engine="newmm")))

X_train, X_test, y_train, y_test = train_test_split(
    df["tokens"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

n_test = len(X_test)


def predict_one_at_a_time(model, X_test_tfidf_matrix):
    """วัด latency แบบ 'ลูกค้า 1 คนพิมพ์มา 1 ข้อความ ต้องรอกี่ ms' จริง ๆ — เรียก .predict()
    ทีละแถวแยกกัน ไม่ batch ทั้งชุดพร้อมกันแล้วหารเฉลี่ย เพราะ batch predict จะเอา fixed overhead
    (เรียกเข้า C extension/คำนวณ kernel matrix ฯลฯ) ไปเฉลี่ยทับทุกแถว ทำให้ตัวเลขต่อแถวต่ำกว่า
    latency ที่ request จริงทีละ 1 ข้อความจะเจอมาก (ดู demo UI ที่ 4-demo-ui/app.py เทียบ)"""
    preds = []
    t0 = time.perf_counter()
    for i in range(X_test_tfidf_matrix.shape[0]):
        preds.append(model.predict(X_test_tfidf_matrix[i])[0])
    elapsed_ms = (time.perf_counter() - t0) / X_test_tfidf_matrix.shape[0] * 1000
    return preds, elapsed_ms


svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train_tfidf, y_train)
svm_preds, svm_infer_ms = predict_one_at_a_time(svm, X_test_tfidf)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)
rf_preds, rf_infer_ms = predict_one_at_a_time(rf, X_test_tfidf)

model_predictions = {
    "classical_svm": svm_preds,
    "classical_rf": rf_preds,
}
infer_time_ms_per_sample = {
    "classical_svm": svm_infer_ms,
    "classical_rf": rf_infer_ms,
}

# --- load Track B (WangchanBERTa) predictions if available ---
bert_cache = ROOT / "2-train-test-model" / "wangchanberta" / "preds_cache.pkl"
if bert_cache.exists():
    with open(bert_cache, "rb") as f:
        cached = pickle.load(f)
    assert list(cached["y_test"]) == list(y_test), "BERT predictions used a different split"
    model_predictions["wangchanberta"] = cached["bert_preds"]
    infer_time_ms_per_sample["wangchanberta"] = cached.get("infer_time_ms_per_sample")
else:
    print(f"[warn] {bert_cache} not found — skipping wangchanberta")

# --- load Track C (LLM prompting) predictions if available ---
llm_cache = ROOT / "2-train-test-model" / "llm_prompting" / "preds_cache.pkl"
if llm_cache.exists():
    with open(llm_cache, "rb") as f:
        cached = pickle.load(f)
    assert list(cached["y_test"]) == list(y_test), "LLM predictions used a different split"
    model_predictions["llm_prompting"] = cached["llm_preds"]
    infer_time_ms_per_sample["llm_prompting"] = cached.get("infer_time_ms_per_sample")
else:
    print(f"[warn] {llm_cache} not found — skipping llm_prompting")

results = []
for model_name, preds in model_predictions.items():
    results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "macro_precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "infer_time_ms_per_sample": infer_time_ms_per_sample.get(model_name),
    })

# เรียงตาม track order คงที่ (A: svm, rf -> B: wangchanberta -> C: llm) ไม่เรียงตาม macro_f1
MODEL_ORDER = ["classical_svm", "classical_rf", "wangchanberta", "llm_prompting"]
results_df = pd.DataFrame(results)
results_df["model"] = pd.Categorical(results_df["model"], categories=MODEL_ORDER, ordered=True)
results_df = results_df.sort_values("model").reset_index(drop=True)
print(results_df)
results_df.to_csv(RESULTS_DIR / "model_comparison_results.csv", index=False)

for model_name, preds in model_predictions.items():
    cm = confusion_matrix(y_test, preds, labels=list(LABEL2ID.keys()))
    print(f"\n{model_name} confusion matrix:\n{cm}")

with open(RESULTS_DIR / "preds_cache.pkl", "wb") as f:
    pickle.dump({"y_test": list(y_test), **model_predictions}, f)
