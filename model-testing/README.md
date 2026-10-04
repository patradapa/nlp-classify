# Model Testing Pipeline — Chat Classification (3 Tracks)

เปรียบเทียบ 3 แนวทางสำหรับจำแนกข้อความแชทลูกค้าเข้า 4 หมวดหมู่:
`sale_oneoff` | `sale_enterprise` | `it_support` | `admin_general`

- **Track A** — Classical ML (TF-IDF + SVM / Random Forest)
- **Track B** — Thai BERT (WangchanBERTa fine-tuned)
- **Track C** — LLM Prompting (Typhoon 1B, รัน local ผ่าน HuggingFace transformers)

สเปกต้นฉบับอยู่ที่ [`../model-testing-pipeline-generic-product.md`](../model-testing-pipeline-generic-product.md)

---

## 1. Setup

```bash
cd model-testing
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

> บน macOS (Apple Silicon) ถ้าเจอ error ตอนโหลด tokenizer ของ WangchanBERTa หรือ Typhoon
> ให้เช็คว่าติดตั้ง `sentencepiece`, `protobuf`, `accelerate` ครบ (มีอยู่ใน `requirements.txt` แล้ว)

ข้อมูล mock (`data/mock_conversations_400.csv`, 400 แถว, 100 ต่อหมวดหมู่) แนบมาให้แล้ว
ไม่ต้องรัน generator ซ้ำ เว้นแต่ต้องการปรับจำนวน/เพิ่ม template — ดูหัวข้อ 6

---

## 2. วิธีรันแต่ละ Track

รันทุกคำสั่งจาก root ของ `model-testing/` (ที่ activate venv แล้ว)

### Track A — Classical ML (เบาสุด ไม่ต้องใช้ GPU)

```bash
python -m classical_ml.train_classical
```

เทรน SVM + Random Forest บน TF-IDF features แล้ว print `classification_report` ของทั้งคู่
ใช้เวลาไม่ถึงวินาที ไม่ต้องมี GPU

### Track B — WangchanBERTa (ต้องดาวน์โหลดโมเดลครั้งแรก ~400MB)

```bash
python wangchanberta/train_bert.py
```

Fine-tune `airesearch/wangchanberta-base-att-spm-uncased` เป็น 3 epochs (ลดจาก 5 ตามสเปกต้นฉบับ
เพราะรันบน CPU/MPS ไม่มี CUDA GPU — ถ้ามี GPU แรงกว่านี้ปรับ `num_train_epochs`/`per_device_train_batch_size`
ใน [`wangchanberta/train_bert.py`](wangchanberta/train_bert.py) ได้)

ผลลัพธ์:
- print `classification_report`
- บันทึก predictions ไว้ที่ `wangchanberta/preds_cache.pkl` (ให้ `evaluate_all.py` เอาไปใช้ต่อ)
- เซฟ checkpoint ไว้ที่ `wangchanberta-chat-classifier/checkpoint-*` (ไฟล์ใหญ่ ~1.2GB/checkpoint)

ใช้เวลาเทรนจริง ~40 วินาที (บน MPS, 320 train samples) แต่ครั้งแรกจะช้ากว่านี้เพราะต้องโหลดโมเดลจาก HuggingFace Hub

### Track C — LLM Prompting (Typhoon 1B, local)

> สเปกต้นฉบับใช้ Anthropic API (`ANTHROPIC_API_KEY`) แต่ในเครื่องนี้ไม่มี key เลยเปลี่ยนมาใช้
> `typhoon-ai/llama3.2-typhoon2-1b-instruct` รัน local แทน (ดูเหตุผลและการเปรียบเทียบ resource ด้านล่าง)

```bash
python llm_prompting/classify_llm.py
```

โหลดโมเดล Typhoon 1B (ดาวน์โหลดครั้งแรก ~2.5GB) แล้ว classify ทีละข้อความด้วย few-shot prompting
(zero-shot ไม่ fine-tune) ช้ากว่า Track A/B มาก เพราะเป็น generative model ต้อง generate token ทีละตัว
(~1 วินาที/ข้อความ, รวม ~80 วินาทีสำหรับ test set 80 ตัวอย่าง)

ผลลัพธ์: print `classification_report` + บันทึก `llm_prompting/preds_cache.pkl`

**ถ้ามี `ANTHROPIC_API_KEY`** และอยากกลับไปใช้ Claude API ตามสเปกเดิม แก้ `classify_with_llm()`
ใน [`llm_prompting/classify_llm.py`](llm_prompting/classify_llm.py) ให้เรียก `anthropic.Anthropic().messages.create(...)`
แทนการเรียก local pipeline (โค้ดตัวอย่างอยู่ในสเปก §5)

---

## 3. รวมผลเปรียบเทียบทั้ง 3 Track

**ต้องรัน Track A, B, C ให้ครบก่อน** (แต่ละ track จะเซฟ `preds_cache.pkl` ไว้ในโฟลเดอร์ตัวเอง)
แล้วค่อยรัน:

```bash
python evaluation/evaluate_all.py
```

จะ retrain Track A ใหม่ (เร็วมาก ไม่มีปัญหา) แล้วโหลด predictions ของ Track B/C จาก cache มาเทียบ
output:
- print ตาราง accuracy / macro-F1 + confusion matrix ของทุกโมเดล
- เซฟตารางไว้ที่ `evaluation/model_comparison_results.csv`
- เซฟ `evaluation/preds_cache.pkl` (รวม predictions ทุกโมเดล ไว้ให้ `mcnemar_test.py` ใช้ต่อ)

### ทดสอบนัยสำคัญทางสถิติ (McNemar's test)

```bash
python evaluation/mcnemar_test.py
```

เทียบทุกคู่โมเดลที่มีอยู่ใน `evaluation/preds_cache.pkl` แล้ว print p-value ของแต่ละคู่
(ต้องรัน `evaluate_all.py` ก่อนเสมอ)

---

## 4. เปรียบเทียบ Resource การใช้งาน (เวลา/หน่วยความจำ/ขนาดโมเดล)

รันแยกแต่ละ track เป็น process เดี่ยว (เพื่อวัด peak memory แยกกัน ไม่ปนกัน):

```bash
python evaluation/resource_comparison.py classical
python evaluation/resource_comparison.py wangchanberta   # ต้องรัน Track B (train_bert.py) มาก่อนแล้ว มี checkpoint
python evaluation/resource_comparison.py llm
```

แต่ละคำสั่งจะ print ตาราง (params/ขนาดโมเดล, disk size, load/train/inference time) และเซฟเป็น
`evaluation/resource_<track>.csv` — รวม 3 ไฟล์เข้าด้วยกันด้วย pandas เพื่อดูภาพรวม หรือดูสรุปที่รันไว้แล้วใน
`evaluation/resource_comparison_all.csv`

ถ้าอยากวัด peak memory ของ OS โดยตรง (แม่นกว่า resource module ของ Python) รันผ่าน `/usr/bin/time -l`
(macOS) หรือ `/usr/bin/time -v` (Linux):

```bash
/usr/bin/time -l python evaluation/resource_comparison.py llm
```

**สรุปผลที่วัดได้ (MacBook, 16GB RAM, Apple Silicon MPS):**

| Model | Device | Size | Train | Inference/sample | Peak Memory |
|---|---|---|---|---|---|
| classical_svm | CPU | 0.11 MB | 0.06s | 0.02 ms | 213 MB |
| classical_rf | CPU | 1.2 MB | 0.09s | 0.06 ms | 213 MB |
| wangchanberta | MPS | 1,263 MB | 42.5s | 3.05 ms | 1,542 MB |
| llm_prompting (typhoon-1b) | MPS | 2,472 MB | — (no fine-tune) | 954.8 ms | 3,790 MB |

Classical ML เบาที่สุดในทุกมิติ, WangchanBERTa อยู่กลาง, LLM prompting หนักสุด (เพราะเป็น
generative decoding ทีละ token ต่างจาก classification head ที่ forward pass ครั้งเดียว)

---

## 5. ลำดับการรันแบบเต็ม (ครั้งแรก)

```bash
source venv/bin/activate
python -m classical_ml.train_classical
python wangchanberta/train_bert.py
python llm_prompting/classify_llm.py
python evaluation/evaluate_all.py
python evaluation/mcnemar_test.py
```

---

## 6. ปรับขนาด/เพิ่มความหลากหลายของ mock data

แก้ `count` (ปัจจุบัน 100 ต่อหมวดหมู่) หรือเพิ่ม template ในตัวแปร `*_BODIES` ใน
[`data/generate_mock_data.py`](data/generate_mock_data.py) แล้วรันใหม่:

```bash
cd data
python generate_mock_data.py
```

จะ overwrite `mock_conversations_400.csv` เดิม (สุ่มด้วย `seed=42` เหมือนเดิม ผลลัพธ์ reproducible)
เมื่อมีข้อมูลจริงจากลูกค้า ให้แทนที่ไฟล์ CSV นี้ด้วยข้อมูลจริงแล้วรันทั้ง pipeline (หัวข้อ 5) ซ้ำ

---

## 7. Known Issues / ข้อควรระวัง

- **Track A ได้ accuracy 100%** — เพราะ mock data เป็น template ที่แยก vocabulary กันชัดเจนเกินจริง
  ไม่ใช่ตัวแทน generalization ที่แท้จริง ต้องทดสอบกับข้อมูลลูกค้าจริงถึงจะสรุปได้แม่นยำ
- **Track C (Typhoon 1B)** เป็น instruct model ขนาดเล็ก บางครั้ง echo ข้อความกลับมาแทนที่จะตอบ label
  ที่กำหนด — โค้ดปัจจุบันมี prompt ที่เข้มงวด (ห้าม copy ข้อความ, บังคับ label ต้องตรงกับ 4 ค่าที่กำหนด)
  ถ้ายังเจอ parse error บ่อย ให้ปรับ `SYSTEM_PROMPT`/`FEW_SHOT_EXAMPLES` ใน `llm_prompting/classify_llm.py`
- **wangchanberta-chat-classifier/** (ไฟล์ checkpoint ที่ train_bert.py สร้าง) มีขนาดใหญ่ (~1.2GB ต่อ epoch)
  ควร `.gitignore` ไม่ commit เข้า git
