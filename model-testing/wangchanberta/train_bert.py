import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
LABEL2ID = {"sale_oneoff": 0, "sale_enterprise": 1, "it_support": 2, "admin_general": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

df = pd.read_csv("data/mock_conversations_400.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

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


train_dataset = ChatDataset(X_train, y_train)   # ใช้ text ดิบ (ไม่ต้อง tokenize ล่วงหน้าแบบ Track A)
test_dataset = ChatDataset(X_test, y_test)

# หมายเหตุ: ลด epochs/batch size ลงจากสเปกต้นฉบับ (5 epochs) เพราะรันบน CPU/MPS ไม่มี CUDA GPU
args = TrainingArguments(
    output_dir="./wangchanberta-chat-classifier",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=[],
)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset)

if __name__ == "__main__":
    trainer.train()

    bert_preds_ids = trainer.predict(test_dataset).predictions.argmax(axis=1)
    bert_preds = [ID2LABEL[i] for i in bert_preds_ids]

    from sklearn.metrics import classification_report
    print("WangchanBERTa:\n", classification_report(y_test, bert_preds))

    import pickle
    with open("wangchanberta/preds_cache.pkl", "wb") as f:
        pickle.dump({"y_test": list(y_test), "bert_preds": bert_preds}, f)
