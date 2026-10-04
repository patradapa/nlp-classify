# Spec: ทดสอบ Model 3 Track — Chat Classification (Multi-Model AI Chat App)

> เอกสารนี้ใช้สำหรับส่งต่อให้ coding agent / dev เริ่มเขียนโค้ดทดสอบ/เปรียบเทียบ 3 โมเดล:
> Classical ML (TF-IDF + SVM/Random Forest), Thai BERT (WangchanBERTa fine-tuned), LLM Prompting
>
> **บริบทสินค้า (mock):** แอป Chat AI ที่รวมหลายโมเดล (เช่น GPT, Claude, Gemini, Llama) ไว้ในที่เดียว
> ให้ผู้ใช้เลือกคุยกับโมเดลไหนก็ได้ — ข้อมูล mockup ทั้งหมดในไฟล์นี้คือแชทของ "ลูกค้าของสินค้านี้เอง"
> (ไม่เกี่ยวกับโรงงานหรือพลังงานใดๆ)

---

## 1. เป้าหมายการทดสอบ

รันข้อความชุดเดียวกันผ่านทั้ง 3 โมเดล แล้วเปรียบเทียบ accuracy, macro-F1, confusion matrix
และเวลาที่ใช้ประมวลผล เพื่อหาว่าแนวทางไหนเหมาะกับข้อมูลแชทจริงที่สุด

**4 หมวดหมู่:** `sale_oneoff` | `sale_enterprise` | `it_support` | `admin_general`

---

## 2. Mockup Dataset — 100 ตัวอย่างต่อหมวดหมู่ (รวม 400 แถว)

เพราะ 400 แถวยาวเกินจะ hardcode ในเอกสารตรงๆ ให้สร้างด้วยสคริปต์ generator แทน (เขียนครั้งเดียว
รันซ้ำได้ ปรับ template เพิ่มความหลากหลายได้ง่ายกว่าแก้ CSV มือ) — รันสคริปต์ด้านล่างจะได้ไฟล์
`mock_conversations_400.csv` ที่มี**ครบ 100 แถวต่อหมวดหมู่จริง** (สุ่มด้วย `seed=42` ทำให้ reproducible)

### 2.1 ตัวอย่างผลลัพธ์ (preview 6 แถวแรกจากที่รันจริง)

```csv
id,text,label
1,สวัสดีค่ะ สลับโมเดลจาก Llama ไปอีกตัวไม่ได้ครับ กดแล้วไม่มีปฏิกิริยา,it_support
2,เรียนทีมงานค่ะ เราสนใจ white-label solution เพื่อนำไปใช้ในผลิตภัณฑ์ของบริษัทเราค่ะ,sale_enterprise
3,สวัสดีครับ ขอเปลี่ยนวิธีการชำระเงินเป็นบัตรใบใหม่ครับ,admin_general
4,ขอถามอะไรหน่อยครับ ขอเปลี่ยนวิธีการชำระเงินเป็นบัตรใบใหม่ครับ,admin_general
5,สวัสดีครับ องค์กรเรามี 5 แผนก อยากทำสัญญารวมบิลเดียวได้ไหมครับ,sale_enterprise
6,เราต้องการ API access แบบ custom rate limit สำหรับใช้ในองค์กรค่ะ,sale_enterprise
```

### 2.2 Generator Script — `generate_mock_data.py`

```python
import csv
import random

random.seed(42)

GREETINGS = [
    "สวัสดีค่ะ", "สวัสดีครับ", "รบกวนสอบถามหน่อยค่ะ", "หวัดดีครับ",
    "ขอโทษที่รบกวนค่ะ", "เรียนทีมงานค่ะ", "สอบถามหน่อยนะคะ", "ขอถามอะไรหน่อยครับ",
    "แอดมินคะ", "รบกวนหน่อยนะครับ",
]

PLANS = ["Starter", "Basic", "Pro", "Individual", "Lite", "Personal"]
SEAT_COUNTS = [5, 8, 10, 12, 15, 20, 25, 30, 50, 100]
CONTRACT_YEARS = [1, 2, 3]
MODELS = ["GPT", "Claude", "Gemini", "Llama", "Mistral"]

SALE_ONEOFF_BODIES = [
    "อยากทราบราคาแพ็กเกจ {plan} สำหรับใช้คนเดียวค่ะ",
    "สนใจสมัครแพ็กเกจ {plan} แบบรายเดือน มีส่วนลดผู้ใช้ใหม่ไหมคะ",
    "ขอทดลองใช้ฟรีก่อนตัดสินใจซื้อแพ็กเกจ {plan} ได้ไหมครับ",
    "อยากซื้อแพ็กเกจ {plan} ครั้งเดียวใช้ยาวๆ มีแบบนี้ไหมคะ",
    "แพ็กเกจ {plan} ใช้โมเดล AI ได้กี่ตัวคะ",
    "ถ้าซื้อแพ็กเกจ {plan} วันนี้ จ่ายรอบเดียวได้ไหมครับ ไม่เอาแบบ subscription",
    "อยากอัปเกรดจาก free trial เป็นแพ็กเกจ {plan} ค่ะ ต้องทำยังไง",
    "ขอเปรียบเทียบแพ็กเกจ {plan} กับแพ็กเกจอื่นหน่อยครับ ใช้คนเดียวพอไหม",
    "สนใจซื้อ credit เพิ่มสำหรับแพ็กเกจ {plan} แบบใช้ครั้งเดียวค่ะ",
    "มีโปรโมชั่นลดราคาแพ็กเกจ {plan} ไหมคะ สนใจซื้อตอนนี้เลย",
    "แพ็กเกจ {plan} ใช้กับมือถือได้ไหมคะ ซื้อไปใช้คนเดียว",
    "อยากซื้อแพ็กเกจ {plan} เป็นของขวัญให้เพื่อนครับ ทำได้ไหม",
]

SALE_ENTERPRISE_BODIES = [
    "บริษัทเรามีพนักงาน {n} คน สนใจแพ็กเกจ Enterprise สำหรับทั้งทีมครับ",
    "อยากทราบเงื่อนไข SLA สำหรับลูกค้าองค์กรขนาดใหญ่ค่ะ",
    "ฝ่ายจัดซื้อขอนัดประชุมเจรจาสัญญาระยะยาว {years} ปีครับ",
    "เราต้องการ API access แบบ custom rate limit สำหรับใช้ในองค์กรค่ะ",
    "สนใจ volume license สำหรับ {n} account ครับ ขอใบเสนอราคาหน่อย",
    "อยากให้ทีมช่วย integrate ระบบเข้ากับ internal tool ของบริษัทเราค่ะ",
    "ผู้บริหารขอดู roadmap สินค้าก่อนเซ็นสัญญาระดับองค์กรครับ",
    "บริษัทเราต้องการ dedicated support และ private model deployment ค่ะ",
    "ขอทราบราคาแบบ annual contract สำหรับ {n} seat ครับ",
    "เราสนใจ white-label solution เพื่อนำไปใช้ในผลิตภัณฑ์ของบริษัทเราค่ะ",
    "องค์กรเรามี {n} แผนก อยากทำสัญญารวมบิลเดียวได้ไหมครับ",
    "ขอคุยเรื่อง enterprise pricing กับทีม sales โดยตรงได้ไหมคะ สัญญา {years} ปี",
]

IT_SUPPORT_BODIES = [
    "ล็อกอินเข้าแอปไม่ได้ค่ะ ขึ้น error ตลอด",
    "โมเดล {model} ตอบช้ามากตั้งแต่เมื่อเช้าครับ",
    "แชทค้าง ส่งข้อความไปแล้วไม่มีการตอบกลับเลยค่ะ",
    "สลับโมเดลจาก {model} ไปอีกตัวไม่ได้ครับ กดแล้วไม่มีปฏิกิริยา",
    "API key ใช้ไม่ได้ค่ะ ขึ้น 401 Unauthorized",
    "ประวัติแชทของฉันหายไปหมดเลยครับ กู้กลับมาได้ไหม",
    "แอปค้างตอนอัปโหลดไฟล์แนบค่ะ",
    "ระบบแจ้ง error ทุกครั้งที่ลองสร้าง workspace ใหม่ครับ",
    "โควต้าการใช้งานไม่อัปเดตทั้งที่จ่ายเงินไปแล้วค่ะ",
    "แอปเด้งออกทุกครั้งที่เปิดบนมือถือครับ",
    "โมเดล {model} ตอบผิดหมวดหมู่ที่เลือกไว้ตลอดค่ะ",
    "อัปโหลดไฟล์ PDF แล้วแอปไม่ยอมอ่านเนื้อหาให้ครับ",
]

ADMIN_GENERAL_BODIES = [
    "ขอใบเสร็จรับเงินของเดือนที่แล้วหน่อยค่ะ",
    "ขอเปลี่ยนอีเมลที่ใช้ล็อกอินเป็นอีเมลนี้ครับ",
    "อยากยกเลิกการสมัครสมาชิกรายเดือนค่ะ ต้องทำยังไง",
    "ขอเปลี่ยนวิธีการชำระเงินเป็นบัตรใบใหม่ครับ",
    "รบกวนส่งใบกำกับภาษีให้อีกครั้งค่ะ ไฟล์เก่าหาย",
    "อยากทราบว่ารอบตัดบัตรของบัญชีฉันคือวันไหนคะ",
    "ขอเพิ่มผู้ใช้ในบัญชีอีก 1 คนได้ไหมครับ",
    "ขอเปลี่ยนชื่อบริษัทในใบเสร็จเป็นชื่อนี้ค่ะ",
    "ลืมรหัสผ่าน ขอลิงก์รีเซ็ตหน่อยครับ",
    "อยากปิดบัญชีถาวร ต้องติดต่อใครคะ",
    "ขอเปลี่ยนชื่อบัญชีผู้ใช้เป็นชื่อใหม่ค่ะ",
    "รบกวนขอประวัติการชำระเงินย้อนหลัง 6 เดือนครับ",
]


def gen_rows(bodies, label, count, fill_fn):
    rows = set()
    attempts = 0
    while len(rows) < count and attempts < count * 30:
        attempts += 1
        greeting = random.choice(GREETINGS) if random.random() > 0.15 else ""
        body_template = random.choice(bodies)
        body = fill_fn(body_template)
        text = f"{greeting} {body}".strip()
        rows.add(text)
    return [(text, label) for text in rows][:count]


def fill_oneoff(t):
    return t.format(plan=random.choice(PLANS))


def fill_enterprise(t):
    return t.format(n=random.choice(SEAT_COUNTS), years=random.choice(CONTRACT_YEARS))


def fill_it(t):
    return t.format(model=random.choice(MODELS))


def fill_admin(t):
    return t


all_rows = []
all_rows += gen_rows(SALE_ONEOFF_BODIES, "sale_oneoff", 100, fill_oneoff)
all_rows += gen_rows(SALE_ENTERPRISE_BODIES, "sale_enterprise", 100, fill_enterprise)
all_rows += gen_rows(IT_SUPPORT_BODIES, "it_support", 100, fill_it)
all_rows += gen_rows(ADMIN_GENERAL_BODIES, "admin_general", 100, fill_admin)

random.shuffle(all_rows)

with open("mock_conversations_400.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text", "label"])
    for i, (text, label) in enumerate(all_rows, start=1):
        writer.writerow([i, text, label])

from collections import Counter
print(Counter(label for _, label in all_rows))
print("total:", len(all_rows))
```

รันแล้วได้ผลลัพธ์ยืนยันว่าครบตามสเปก:

```
Counter({'it_support': 100, 'sale_enterprise': 100, 'admin_general': 100, 'sale_oneoff': 100})
total: 400
```

> ไฟล์ `mock_conversations_400.csv` ที่รันได้จริง (400 แถว) แนบมาพร้อมกับเอกสารนี้แล้ว —
> ใช้ไฟล์นี้ตรงๆ ได้เลยโดยไม่ต้องรันสคริปต์ซ้ำ หรือรัน re-run เพื่อ regenerate/ปรับ template เพิ่มก็ได้

---

## 3. Track A — Classical ML (TF-IDF + SVM / Random Forest)

### 3.1 Preprocessing

```python
# classical_ml/preprocessing.py
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
```

### 3.2 Train & Evaluate

```python
# classical_ml/train_classical.py
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train_tfidf, y_train)
svm_preds = svm.predict(X_test_tfidf)
print("SVM:\n", classification_report(y_test, svm_preds))

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_tfidf, y_train)
rf_preds = rf.predict(X_test_tfidf)
print("Random Forest:\n", classification_report(y_test, rf_preds))
```

---

## 4. Track B — Thai BERT (WangchanBERTa fine-tuned)

```python
# wangchanberta/train_bert.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
LABEL2ID = {"sale_oneoff": 0, "sale_enterprise": 1, "it_support": 2, "admin_general": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

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

args = TrainingArguments(
    output_dir="./wangchanberta-chat-classifier",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset)
trainer.train()

bert_preds_ids = trainer.predict(test_dataset).predictions.argmax(axis=1)
bert_preds = [ID2LABEL[i] for i in bert_preds_ids]
```

---

## 5. Track C — LLM Prompting (Zero/Few-shot)

```python
# llm_prompting/classify_llm.py
import json

SYSTEM_PROMPT = """คุณเป็นระบบจำแนกข้อความแชทลูกค้าของแอป Chat AI ที่รวมหลายโมเดล (GPT, Claude, Gemini ฯลฯ)
เข้า 4 หมวดหมู่:
- sale_oneoff: ลูกค้ารายบุคคลสนใจซื้อแพ็กเกจครั้งเดียว/ใช้คนเดียว
- sale_enterprise: ลูกค้าองค์กรต้องการสัญญาระยะยาว/หลาย seat/API แบบ custom
- it_support: ปัญหาการใช้งานแอป เช่น error, ล็อกอินไม่ได้, โมเดลตอบช้า
- admin_general: บิล ใบเสร็จ เปลี่ยนข้อมูลบัญชี ยกเลิกสมาชิก

ตอบกลับเป็น JSON เท่านั้น รูปแบบ: {"label": "...", "confidence": 0.0}"""

FEW_SHOT_EXAMPLES = [
    {"text": "อยากซื้อแพ็กเกจ Pro ครั้งเดียวใช้คนเดียวครับ", "label": "sale_oneoff"},
    {"text": "บริษัทเรามีพนักงาน 50 คน สนใจแพ็กเกจ Enterprise", "label": "sale_enterprise"},
    {"text": "ล็อกอินเข้าแอปไม่ได้ค่ะ ขึ้น error ตลอด", "label": "it_support"},
    {"text": "ขอใบเสร็จเดือนที่แล้วค่ะ", "label": "admin_general"},
]

def build_prompt(text: str) -> str:
    examples_text = "\n".join(
        f'ข้อความ: "{ex["text"]}" → {ex["label"]}' for ex in FEW_SHOT_EXAMPLES
    )
    return f"{examples_text}\n\nข้อความ: \"{text}\" → ?"

def classify_with_llm(text: str, client) -> dict:
    """
    client: instance ของ Anthropic/OpenAI SDK ที่ตั้งค่าไว้แล้ว
    คืนค่า dict {"label": str, "confidence": float}
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(text)}],
    )
    raw = response.content[0].text
    return json.loads(raw)

llm_preds = [classify_with_llm(t, client)["label"] for t in X_test]
```

---

## 6. Evaluation Harness (เทียบทั้ง 3 track)

```python
# evaluation/evaluate_all.py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

model_predictions = {
    "classical_svm": svm_preds,
    "classical_rf": rf_preds,
    "wangchanberta": bert_preds,
    "llm_prompting": llm_preds,
}

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
```

### ทดสอบนัยสำคัญทางสถิติ (McNemar's test)

```python
# evaluation/mcnemar_test.py
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

def build_contingency_table(y_true, preds_a, preds_b):
    correct_a = np.array(preds_a) == np.array(y_true)
    correct_b = np.array(preds_b) == np.array(y_true)
    both_correct = np.sum(correct_a & correct_b)
    only_a = np.sum(correct_a & ~correct_b)
    only_b = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)
    return [[both_correct, only_a], [only_b, both_wrong]]

table = build_contingency_table(y_test, bert_preds, svm_preds)
result = mcnemar(table, exact=True)
print(f"McNemar's test p-value: {result.pvalue}")
```

---

## 7. Folder Structure

```
model-testing/
├── data/
│   ├── generate_mock_data.py
│   └── mock_conversations_400.csv     # generated (400 rows, 100 ต่อหมวดหมู่)
├── classical_ml/
│   ├── preprocessing.py
│   └── train_classical.py
├── wangchanberta/
│   └── train_bert.py
├── llm_prompting/
│   └── classify_llm.py
├── evaluation/
│   ├── evaluate_all.py
│   ├── mcnemar_test.py
│   └── model_comparison_results.csv   # output
└── requirements.txt
```

`requirements.txt`:
```
pythainlp
scikit-learn
transformers
torch
pandas
statsmodels
anthropic
```

---

## 8. Next Steps สำหรับ Coding Agent

1. สร้างโครง folder ตามหัวข้อ 7 แล้ววาง `mock_conversations_400.csv` (แนบมาให้แล้ว) ลงใน `data/`
2. รัน Track A (Classical ML) ให้ผ่านก่อน เพราะเบาสุด ไม่ต้องใช้ GPU และมี 400 samples เพียงพอต่อการเทรนเบื้องต้น
3. รัน Track B (WangchanBERTa) — ถ้าไม่มี GPU ให้ลด `num_train_epochs`/batch size หรือรันบน Colab
4. รัน Track C (LLM Prompting) — ต้องมี API key ของ LLM provider ที่เลือกใช้
5. รวมผลทั้ง 3 track ผ่าน `evaluate_all.py` แล้วดูตาราง comparison
6. รัน `mcnemar_test.py` เทียบคู่โมเดลที่คะแนนใกล้กันที่สุด
7. ถ้าต้องการข้อมูลมากกว่า 100/หมวดหมู่ แก้ `count` ใน `generate_mock_data.py` หรือเพิ่ม template ใน body list เพื่อความหลากหลายมากขึ้น แล้วรันซ้ำ
8. เมื่อมีข้อมูลจริงจากลูกค้าจริงมากพอ แทนที่ mock CSV ด้วยข้อมูลจริงแล้วรันซ้ำทั้ง pipeline
