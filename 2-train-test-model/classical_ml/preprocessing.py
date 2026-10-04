import re
from pathlib import Path

import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]  # classical_ml/ -> 2-train-test-model/ -> repo root

# ตัด emoji ออกก่อน tokenize กัน TF-IDF เก็บ emoji เป็น feature/noise
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, supplemental symbols, etc.
    "\U00002600-\U000027BF"  # misc symbols & dingbats
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0000FE00-\U0000FE0F"  # variation selectors (เช่น ‼️)
    "]+",
    flags=re.UNICODE,
)

# ยุบตัวอักษรที่ซ้ำกันยาวๆ (คำยืดเสียงแบบแชท เช่น "ค้าบบบบ", "ค่ะะะะ") ให้เหลือตัวเดียว
# กัน TF-IDF มองคำยืดเสียงแต่ละความยาวเป็นคนละ feature
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

print("[Track A][1/5] โหลดข้อมูล mock จาก 1-data/mock_conversations_400.csv ...")
df = pd.read_csv(ROOT / "1-data" / "mock_conversations_400.csv")
print(f"  โหลดแล้ว {len(df)} แถว")

print("[Track A][2/5] ทำความสะอาดข้อความ (ตัด emoji, ยุบตัวอักษรซ้ำ, รวม whitespace) ...")
df["text"] = df["text"].str.replace(EMOJI_PATTERN, "", regex=True)
df["text"] = df["text"].str.replace(REPEATED_CHAR_PATTERN, r"\1", regex=True)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

print("[Track A][3/5] tokenize ด้วย pythainlp (engine=newmm) ...")
df["tokens"] = df["text"].apply(lambda x: " ".join(word_tokenize(x, engine="newmm")))

print("[Track A][4/5] แบ่ง train/test (stratified, 80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(
    df["tokens"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)
print(f"  train={len(X_train)} / test={len(X_test)}")

print("[Track A][5/5] สร้าง TF-IDF features (ngram 1-2, max_features=5000) ...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"  feature matrix: train={X_train_tfidf.shape}, test={X_test_tfidf.shape}")
