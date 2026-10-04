# Model Testing Pipeline — Chat Classification (3 Tracks)

เปรียบเทียบ 3 แนวทางสำหรับจำแนกข้อความแชทลูกค้าเข้า 4 หมวดหมู่:
`sale_oneoff` | `sale_enterprise` | `it_support` | `admin_general`

- **Track A** — Classical ML (TF-IDF + SVM / Random Forest) — [`2-train-test-model/classical_ml/`](2-train-test-model/classical_ml/)
- **Track B** — Thai BERT (WangchanBERTa fine-tuned) — [`2-train-test-model/wangchanberta/`](2-train-test-model/wangchanberta/)
- **Track C** — LLM Prompting (Typhoon 1B, รัน local ผ่าน HuggingFace transformers) — [`2-train-test-model/llm_prompting/`](2-train-test-model/llm_prompting/)

ประวัติ prompt/คำสั่งที่ทำให้เกิดโปรเจคนี้ขึ้นมา (requirement แต่ละรอบที่นำไปสู่โครงสร้างปัจจุบัน) สรุปไว้ที่
[`PROMPTS.md`](PROMPTS.md)

## โครงสร้างโปรเจค

```
.
├── README.md
├── requirements.txt
├── 1-data/
│   ├── generate_mock_data.py       # สร้าง/regenerate mock_conversations_400.csv
│   └── mock_conversations_400.csv  # 400 แถว, ~100 ต่อหมวดหมู่
├── 2-train-test-model/
│   ├── classical_ml/                # Track A
│   │   ├── preprocessing.py
│   │   └── train_classical.py
│   ├── wangchanberta/                # Track B
│   │   └── train_bert.py
│   └── llm_prompting/                # Track C
│       └── classify_llm.py
├── 3-evaluate/
│   ├── evaluate_all.py              # รวมผล accuracy/precision/recall/F1/latency/confusion matrix ทุก track
│   └── results/                      # output ทั้งหมด (CSV/pkl) แยกจากโค้ด — regenerate ได้ด้วยสคริปต์ข้างบน
│       ├── model_comparison_results.csv
│       └── preds_cache.pkl
└── 4-demo-ui/                         # เว็บ UI เดี่ยว (Streamlit) กรอกข้อความแล้วดูผลทั้ง 4 โมเดล (หัวข้อ 5)
    └── app.py
```

ทุกสคริปต์ anchor path ด้วย `Path(__file__)` ภายใน ไม่ได้พึ่ง current working directory
ดังนั้น**รันจากที่ไหนก็ได้** (เช่น `python 2-train-test-model/wangchanberta/train_bert.py` จาก root ของ repo)

---

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

บน Windows ให้ activate แบบนี้แทน (ขั้นตอนอื่นเหมือนกัน):

```powershell
python -m venv venv
venv\Scripts\activate         # Command Prompt / PowerShell
```

ข้อมูล mock (`1-data/mock_conversations_400.csv`, 400 แถว, ~100 ต่อหมวดหมู่) แนบมาให้แล้ว
ไม่ต้องรัน generator ซ้ำ เว้นแต่ต้องการปรับจำนวน/เพิ่ม template — ดูหัวข้อ 6

---

## 2. วิธีรันแต่ละ Track

รันทุกคำสั่งจาก root ของ repo (ที่ activate venv แล้ว)

### Track A — Classical ML (ไม่ต้องใช้ GPU)

```bash
python 2-train-test-model/classical_ml/train_classical.py
```

เทรน SVM + Random Forest บน TF-IDF features แล้ว print `classification_report` ของทั้งคู่
ใช้เวลาไม่ถึงวินาที ไม่ต้องมี GPU

### Track B — WangchanBERTa (ต้องดาวน์โหลดโมเดลครั้งแรก ~400MB)

```bash
python 2-train-test-model/wangchanberta/train_bert.py
```

ผลลัพธ์:
- print `classification_report`
- บันทึก predictions + `infer_time_ms_per_sample` ไว้ที่ `2-train-test-model/wangchanberta/preds_cache.pkl`
  (ให้ `evaluate_all.py` เอาไปใช้ต่อ)
- เซฟ checkpoint ไว้ที่ `2-train-test-model/wangchanberta/wangchanberta-chat-classifier/checkpoint-*` (ไฟล์ใหญ่ ~1.2GB/checkpoint)

ใช้เวลาเทรนจริง ~40 วินาที (320 train samples) แต่ครั้งแรกจะช้ากว่านี้เพราะต้องโหลดโมเดลจาก HuggingFace Hub

### Track C — LLM Prompting (Typhoon 1B, local)

```bash
python 2-train-test-model/llm_prompting/classify_llm.py
```

โหลดโมเดล Typhoon 1B (ดาวน์โหลดครั้งแรก ~2.5GB) แล้ว classify ทีละข้อความด้วย few-shot prompting
(4-shot, ไม่ fine-tune โมเดล) ช้ากว่า Track A/B มาก เพราะเป็น generative model ต้อง generate token ทีละตัว
(~1 วินาที/ข้อความ, รวม ~80 วินาทีสำหรับ test set 80 ตัวอย่าง)

ผลลัพธ์: print `classification_report` + บันทึก predictions + `infer_time_ms_per_sample` ไว้ที่
`2-train-test-model/llm_prompting/preds_cache.pkl`

---

## 3. รวมผลเปรียบเทียบทั้ง 3 Track

**ต้องรัน Track A, B, C ให้ครบก่อน** (แต่ละ track จะเซฟ `preds_cache.pkl` ไว้ในโฟลเดอร์ตัวเอง)
แล้วค่อยรัน:

```bash
python 3-evaluate/evaluate_all.py
```

จะ retrain Track A ใหม่ (เร็วมาก ไม่มีปัญหา, วัด inference latency จากการรันนี้เลย) แล้วโหลด predictions
+ latency ของ Track B/C จาก cache มาเทียบ (latency ของ B/C มาจากตอนที่ `train_bert.py`/`classify_llm.py`
รันจริงครั้งล่าสุด ไม่ได้ rerun ซ้ำที่นี่ เพราะ Track B/C ใช้เวลานาน/โหลดโมเดลหนัก)

output:
- print ตาราง `accuracy` / `macro_precision` / `macro_recall` / `macro_f1` / **`infer_time_ms_per_sample`**
  + confusion matrix ของทุกโมเดล
- Save ตารางไว้ที่ `3-evaluate/results/model_comparison_results.csv`
- Save `3-evaluate/results/preds_cache.pkl` (รวม predictions ทุกโมเดล เผื่อเอาไปวิเคราะห์เพิ่ม)

---

## 4. ลำดับการรันแบบเต็ม (ครั้งแรก)

```bash
source venv/bin/activate      # macOS/Linux
python 2-train-test-model/classical_ml/train_classical.py
python 2-train-test-model/wangchanberta/train_bert.py
python 2-train-test-model/llm_prompting/classify_llm.py
python 3-evaluate/evaluate_all.py
```

บน Windows ให้ activate แบบนี้แทน (ขั้นตอนอื่นเหมือนกัน):

```powershell
venv\Scripts\activate         # Command Prompt / PowerShell
```

ถ้าอยากล้างทุกอย่างที่เกิดจากการรัน (checkpoint, prediction cache, ผล evaluate — ไม่ลบ mock data CSV)
แล้วเริ่มใหม่ตั้งแต่ต้น ใช้ [`reset_pipeline.py`](reset_pipeline.py):

```bash
python reset_pipeline.py --dry-run   # ดูก่อนว่าจะลบอะไร
python reset_pipeline.py             # ลบจริง (ถามยืนยันก่อน)
```

---

## 5. Demo UI — ทดลองพิมพ์ข้อความแล้วดูผลทั้ง 4 โมเดล

**ต้องรัน Track B (`train_bert.py`) มาก่อนแล้วมี checkpoint** (ไม่งั้น Track B จะโชว์ N/A ใน UI)

```bash
streamlit run 4-demo-ui/app.py
```

browser จะเปิดอัตโนมัติที่ `http://localhost:8501` — มี text area ให้กรอกข้อความหรือบทสนทนาหลายข้อความ
(เช่น `ลูกค้า: ...` ขึ้นบรรทัดใหม่แล้วตามด้วย `แอดมิน: ...`) กดปุ่ม "ตรวจสอบ" แล้วหน้าเว็บจะเรียก
SVM/RF/WangchanBERTa/Typhoon-1B ทั้ง 4 ตัวจริง ๆ (ไม่ใช่ mock) แล้วโชว์หมวดหมู่ที่ทำนายได้ + เวลาที่ใช้ (ms)
ของแต่ละโมเดลเทียบกันในตารางเดียว — มีปุ่ม "สุ่มตัวอย่างข้อความ" ให้ลองได้ทันทีโดยไม่ต้องพิมพ์เอง

**หมายเหตุ:**
- โมเดลทั้งหมดโหลด**ครั้งเดียว** ด้วย `@st.cache_resource` ไม่ใช่ทุกครั้งที่กดปุ่ม (Streamlit rerun สคริปต์
  ทั้งไฟล์ทุกครั้งที่มี interaction ถ้าไม่ cache จะโหลด Typhoon-1B ~2.5GB ใหม่ทุกคลิก) — รอบแรกที่เปิดหน้าเว็บ
  จะใช้เวลาสักพัก (เทรน SVM/RF + โหลด WangchanBERTa checkpoint + โหลด Typhoon-1B)

---

## 6. ปรับขนาด/เพิ่มความหลากหลายของ mock data

แก้ `count` (ปัจจุบัน 100 ต่อหมวดหมู่) หรือเพิ่ม template ในตัวแปร `*_BODIES` ใน
[`1-data/generate_mock_data.py`](1-data/generate_mock_data.py) แล้วรันใหม่:

```bash
python 1-data/generate_mock_data.py
```

จะ overwrite `1-data/mock_conversations_400.csv` เดิม (สุ่มด้วย `seed=42` เหมือนเดิม ผลลัพธ์ reproducible)
เมื่อมีข้อมูลจริงจากลูกค้า ให้แทนที่ไฟล์ CSV นี้ด้วยข้อมูลจริงแล้วรันทั้ง pipeline (หัวข้อ 4) ซ้ำ

Mock data ปัจจุบันมี noise ที่จงใจเพิ่มเพื่อความสมจริง:
- emoji และคำลงท้ายแบบยืดเสียง (`ครับบบบ`, `ค่าบ` ฯลฯ) แจกแบบสุ่มเท่ากันทุก label
- บางแถวเป็นบทสนทนาหลายข้อความ (ลูกค้า/แอดมิน) ไม่ใช่ประโยคเดียว
- ประโยคกำกวมที่คาบเกี่ยวคำศัพท์กับหมวดข้างเคียง (เช่น sale_oneoff ↔ sale_enterprise)
- label noise ~6% สลับไปหมวดที่คาบเกี่ยวกัน จำลองความผิดพลาดของ human annotator
