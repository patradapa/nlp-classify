import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ต้อง seed ก่อนโหลดโมเดล: WangchanBERTa
# from_pretrained() — ถ้าไม่ seed ก่อนจุดนี้ ทุกรอบที่รันจะได้ initial weights คนละชุด เทรนออกมา
# ได้โมเดล/ผลลัพธ์ไม่เหมือนกันทุกครั้ง
set_seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent             # 2-train-test-model/wangchanberta/
ROOT = SCRIPT_DIR.parent.parent                          # repo root

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
LABEL2ID = {"sale_oneoff": 0, "sale_enterprise": 1, "it_support": 2, "admin_general": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

print("[Track B][1/6] โหลดข้อมูล mock จาก 1-data/mock_conversations_400.csv ...")
df = pd.read_csv(ROOT / "1-data" / "mock_conversations_400.csv")
print(f"  โหลดแล้ว {len(df)} แถว")

print("[Track B][2/6] แบ่ง train/test (stratified, 80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)
print(f"  train={len(X_train)} / test={len(X_test)}")

print(f"[Track B][3/6] โหลด tokenizer + model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=4, label2id=LABEL2ID, id2label=ID2LABEL
)


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=256)
        self.labels = [LABEL2ID[l] for l in labels]

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


print("[Track B][4/6] tokenize train/test set (max_length=256) ...")
train_dataset = ChatDataset(X_train, y_train)   # ใช้ raw text (ไม่ต้อง tokenize ล่วงหน้าแบบ Track A)
test_dataset = ChatDataset(X_test, y_test)

args = TrainingArguments(
    output_dir=str(SCRIPT_DIR / "wangchanberta-chat-classifier"),
    seed=42,  # explicit
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=[],
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset,
    processing_class=tokenizer,
)

if __name__ == "__main__":
    print(f"[Track B][5/6] เริ่มเทรน (fine-tune {args.num_train_epochs} epochs) ...")
    trainer.train()

    print("[Track B][6/6] ทำนายบน test set + บันทึก cache ...")
    bert_preds_ids = trainer.predict(test_dataset).predictions.argmax(axis=1)
    bert_preds = [ID2LABEL[i] for i in bert_preds_ids]

    # วัด latency แบบ "ลูกค้า 1 คนพิมพ์มา 1 ข้อความ ต้องรอกี่ ms" จริง ๆ — forward pass ทีละ 1 ข้อความ
    model.eval()
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    with torch.no_grad():
        for text in X_test:
            enc = tokenizer([text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
            model(**enc)
    infer_time_ms_per_sample = (time.perf_counter() - t0) / len(X_test) * 1000

    print("WangchanBERTa:\n", classification_report(y_test, bert_preds))

    with open(SCRIPT_DIR / "preds_cache.pkl", "wb") as f:
        pickle.dump({
            "y_test": list(y_test),
            "bert_preds": bert_preds,
            "infer_time_ms_per_sample": infer_time_ms_per_sample,
        }, f)
    print(f"  บันทึก predictions ไว้ที่ {SCRIPT_DIR / 'preds_cache.pkl'}")
