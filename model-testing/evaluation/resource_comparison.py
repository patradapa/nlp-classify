import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_split():
    df = pd.read_csv("data/mock_conversations_400.csv")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    return X_train_raw, X_test_raw, y_train, y_test


def run_classical():
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from pythainlp.tokenize import word_tokenize

    X_train_raw, X_test_raw, y_train, y_test = load_split()
    X_train = X_train_raw.apply(lambda x: " ".join(word_tokenize(x, engine="newmm")))
    X_test = X_test_raw.apply(lambda x: " ".join(word_tokenize(x, engine="newmm")))

    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    tfidf_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(X_train_tfidf, y_train)
    svm_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    svm.predict(X_test_tfidf)
    svm_infer_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_tfidf, y_train)
    rf_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    rf.predict(X_test_tfidf)
    rf_infer_time = time.perf_counter() - t0

    svm_size = len(pickle.dumps(svm)) + len(pickle.dumps(vectorizer))
    rf_size = len(pickle.dumps(rf)) + len(pickle.dumps(vectorizer))
    n = len(X_test)

    result = [
        {
            "model": "classical_svm", "device": "cpu",
            "params_or_size": f"{sum(svm.n_support_)} support vectors",
            "disk_size_mb": round(svm_size / 1e6, 2),
            "load_time_sec": 0.0,
            "train_time_sec": round(tfidf_time + svm_train_time, 3),
            "infer_time_sec_80samples": round(svm_infer_time, 4),
            "infer_time_ms_per_sample": round(svm_infer_time / n * 1000, 3),
        },
        {
            "model": "classical_rf", "device": "cpu",
            "params_or_size": "200 trees",
            "disk_size_mb": round(rf_size / 1e6, 2),
            "load_time_sec": 0.0,
            "train_time_sec": round(tfidf_time + rf_train_time, 3),
            "infer_time_sec_80samples": round(rf_infer_time, 4),
            "infer_time_ms_per_sample": round(rf_infer_time / n * 1000, 3),
        },
    ]
    return result


def run_wangchanberta():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _, X_test_raw, _, y_test = load_split()
    n = len(X_test_raw)

    ckpt_dirs = sorted(Path("wangchanberta-chat-classifier").glob("checkpoint-*"))
    if not ckpt_dirs:
        raise SystemExit("no wangchanberta checkpoint found — run wangchanberta/train_bert.py first")
    bert_ckpt = str(ckpt_dirs[-1])
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(bert_ckpt).to(device)
    model.eval()
    load_time = time.perf_counter() - t0

    params = model.num_parameters()
    disk_mb = sum(f.stat().st_size for f in Path(bert_ckpt).glob("**/*") if f.is_file()) / 1e6

    t0 = time.perf_counter()
    with torch.no_grad():
        enc = tokenizer(list(X_test_raw), truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
        _ = model(**enc)
    infer_time = time.perf_counter() - t0

    return [{
        "model": "wangchanberta", "device": device,
        "params_or_size": f"{params:,} params",
        "disk_size_mb": round(disk_mb, 2),
        "load_time_sec": round(load_time, 3),
        "train_time_sec": 42.54,  # measured separately via Trainer.train() (3 epochs, 320 train samples)
        "infer_time_sec_80samples": round(infer_time, 4),
        "infer_time_ms_per_sample": round(infer_time / n * 1000, 3),
    }]


def run_llm(n_sample=20):
    import torch
    from transformers import pipeline
    from llm_prompting.classify_llm import classify_with_llm, MODEL_NAME

    _, X_test_raw, _, y_test = load_split()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.perf_counter()
    pipe = pipeline("text-generation", model=MODEL_NAME, dtype=torch.bfloat16, device=device)
    load_time = time.perf_counter() - t0

    params = pipe.model.num_parameters()
    disk_mb = 2471645608 / 1e6  # model.safetensors, from HF content-addressed blob store

    sample = list(X_test_raw)[:n_sample]
    t0 = time.perf_counter()
    for t in sample:
        try:
            classify_with_llm(t, pipe)
        except Exception:
            pass
    infer_time_sample = time.perf_counter() - t0

    return [{
        "model": "llm_prompting (typhoon-1b, local)", "device": device,
        "params_or_size": f"{params:,} params",
        "disk_size_mb": round(disk_mb, 2),
        "load_time_sec": round(load_time, 3),
        "train_time_sec": 0.0,  # zero/few-shot prompting, no fine-tuning
        "infer_time_sec_80samples": round(infer_time_sample / n_sample * 80, 4),
        "infer_time_ms_per_sample": round(infer_time_sample / n_sample * 1000, 3),
    }]


TRACKS = {"classical": run_classical, "wangchanberta": run_wangchanberta, "llm": run_llm}
# หมายเหตุ: รันแต่ละ track แยก process ด้วย `/usr/bin/time -l` (macOS) เพื่อวัด
# peak RSS / peak memory footprint แบบแยกโมเดล ไม่ปนกัน แล้วรวมผลด้วย pandas.concat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("track", choices=TRACKS.keys())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = TRACKS[args.track]()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_path = args.out or f"evaluation/resource_{args.track}.csv"
    df.to_csv(out_path, index=False)
