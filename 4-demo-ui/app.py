"""
Demo UI (Streamlit) — กรอกข้อความ/บทสนทนา 1 ครั้ง แล้วดูว่าทั้ง 4 โมเดล (SVM, Random Forest,
WangchanBERTa, Typhoon-1B) ทำนายเป็นหมวดไหน และใช้เวลากี่ ms ต่อโมเดล เทียบกันในหน้าเดียว

รัน:
    streamlit run 4-demo-ui/app.py
แล้ว browser จะเปิดให้อัตโนมัติที่ http://localhost:8501

หมายเหตุ:
- โมเดลโหลด "ครั้งเดียว" ด้วย @st.cache_resource (ไม่ใช่ทุกครั้งที่กดปุ่ม — Streamlit rerun ทั้งสคริปต์
  ทุกครั้งที่มี interaction ถ้าไม่ cache จะโหลด Typhoon-1B ~2.5GB ใหม่ทุกคลิก)
- ถ้า WangchanBERTa ยังไม่มี checkpoint (ยังไม่ได้รัน train_bert.py) หรือ Typhoon โหลดไม่ผ่าน
  จะ skip track นั้น แล้วโชว์ "N/A" ในตาราง ไม่ทำให้แอปล้ม
"""

import re
import sys
import time
from pathlib import Path

import streamlit as st
from pythainlp.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2-train-test-model"))  # ให้ import classical_ml/wangchanberta/llm_prompting ได้

WHITESPACE_PATTERN = re.compile(r"\s+")

LABEL_DESC = {
    "sale_oneoff": "ลูกค้ารายบุคคล ซื้อครั้งเดียว/ใช้คนเดียว",
    "sale_enterprise": "ลูกค้าองค์กร สัญญาระยะยาว/หลาย seat",
    "it_support": "ปัญหาการใช้งานแอป เช่น error, ล็อกอินไม่ได้",
    "admin_general": "บิล ใบเสร็จ บัญชี ยกเลิกสมาชิก",
}

SAMPLES = [
    "อยากทราบราคาแพ็กเกจ Pro สำหรับใช้คนเดียวค่ะ",
    "บริษัทเรามีพนักงาน 50 คน สนใจแพ็กเกจ Enterprise สำหรับทั้งทีมครับ",
    "ล็อกอินเข้าแอปไม่ได้ค่ะ ขึ้น error ตลอด 😅",
    "ขอใบเสร็จรับเงินของเดือนที่แล้วหน่อยค่ะ",
    "ลูกค้า: สวัสดีค่ะ\nแอดมิน: สวัสดีค่ะ มีอะไรให้ช่วยคะ\nลูกค้า: โมเดลตอบช้ามากตั้งแต่เมื่อเช้าครับ",
]


# ---------------------------------------------------------------------------
# Track A: Classical ML — import โมดูลเดิมตรงๆ (ไม่ copy logic ซ้ำ)
# preprocessing.py รัน load+clean+tokenize+TF-IDF ตอน import, train_classical.py fit SVM/RF ตอน import
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังเทรน Track A (SVM + Random Forest) ...")
def load_classical():
    from classical_ml import preprocessing as prep_a
    from classical_ml import train_classical as model_a
    return prep_a, model_a


def _clean_for_classical(text: str, prep_a) -> str:
    text = prep_a.EMOJI_PATTERN.sub("", text)
    text = prep_a.REPEATED_CHAR_PATTERN.sub(r"\1", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def predict_classical(text: str, prep_a, model):
    cleaned = _clean_for_classical(text, prep_a)
    tokens = " ".join(word_tokenize(cleaned, engine="newmm"))
    t0 = time.perf_counter()
    pred = model.predict(prep_a.vectorizer.transform([tokens]))[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return pred, elapsed_ms


# ---------------------------------------------------------------------------
# Track B: WangchanBERTa — โหลด checkpoint ล่าสุด ถ้ามี
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลด Track B (WangchanBERTa) ...")
def load_wangchanberta():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt_root = ROOT / "2-train-test-model" / "wangchanberta" / "wangchanberta-chat-classifier"
    ckpt_dirs = sorted(ckpt_root.glob("checkpoint-*"))
    if not ckpt_dirs:
        return None

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # โหลด tokenizer จาก base model เสมอ (fine-tune ไม่เปลี่ยน vocab/tokenizer, checkpoint เก่าบางตัว
    # ไม่มีไฟล์ tokenizer ติดมาด้วย)
    tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dirs[-1])).to(device)
    model.eval()
    return tokenizer, model, device


def predict_wangchanberta(text: str, bundle):
    import torch

    tokenizer, model, device = bundle
    t0 = time.perf_counter()
    with torch.no_grad():
        enc = tokenizer([text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
        logits = model(**enc).logits
        pred_id = logits.argmax(dim=-1).item()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return model.config.id2label[pred_id], elapsed_ms


# ---------------------------------------------------------------------------
# Track C: Typhoon-1B — few-shot prompting (ไม่ fine-tune) ผ่าน classify_llm.py เดิม
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลด Track C (Typhoon-1B) — รอบแรกอาจใช้เวลาสักพัก ...")
def load_llm():
    import torch
    from transformers import pipeline

    from llm_prompting.classify_llm import MODEL_NAME

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        pipe = pipeline("text-generation", model=MODEL_NAME, dtype=torch.bfloat16, device=device)
        return pipe
    except Exception:
        return None


def predict_llm(text: str, pipe):
    from llm_prompting.classify_llm import classify_with_llm

    t0 = time.perf_counter()
    try:
        label = classify_with_llm(text, pipe)["label"]
    except Exception:
        label = "admin_general"  # fallback เดียวกับใน classify_llm.py
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return label, elapsed_ms


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Chat Intent Classifier — Demo", page_icon="💬")
st.title("Chat Intent Classifier — Demo")
st.caption(
    "กรอกข้อความลูกค้า (ประโยคเดียวหรือบทสนทนาหลายข้อความก็ได้) แล้วดูว่าแต่ละโมเดลทำนายเป็นหมวดไหน ใช้เวลากี่ ms"
)
st.caption("หมวดหมู่ที่รองรับ: `sale_oneoff` · `sale_enterprise` · `it_support` · `admin_general`")

prep_a, model_a = load_classical()
bert_bundle = load_wangchanberta()
llm_pipe = load_llm()

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


def _fill_sample():
    import random
    st.session_state.input_text = random.choice(SAMPLES)


st.text_area("ข้อความ / บทสนทนา", key="input_text", height=150)

col1, col2 = st.columns([1, 1])
submit = col1.button("ตรวจสอบ", type="primary")
col2.button("สุ่มตัวอย่างข้อความ", on_click=_fill_sample)

if submit:
    text = st.session_state.input_text.strip()
    if not text:
        st.warning("กรุณากรอกข้อความก่อน")
    else:
        rows = []

        pred, ms = predict_classical(text, prep_a, model_a.svm)
        rows.append({"Track": "Track A — SVM", "หมวดหมู่": pred, "คำอธิบาย": LABEL_DESC.get(pred, ""), "เวลา (ms)": round(ms, 3)})

        pred, ms = predict_classical(text, prep_a, model_a.rf)
        rows.append({"Track": "Track A — Random Forest", "หมวดหมู่": pred, "คำอธิบาย": LABEL_DESC.get(pred, ""), "เวลา (ms)": round(ms, 3)})

        if bert_bundle is not None:
            pred, ms = predict_wangchanberta(text, bert_bundle)
            rows.append({"Track": "Track B — WangchanBERTa", "หมวดหมู่": pred, "คำอธิบาย": LABEL_DESC.get(pred, ""), "เวลา (ms)": round(ms, 3)})
        else:
            rows.append({"Track": "Track B — WangchanBERTa", "หมวดหมู่": "N/A (ยังไม่มี checkpoint)", "คำอธิบาย": "", "เวลา (ms)": None})

        if llm_pipe is not None:
            pred, ms = predict_llm(text, llm_pipe)
            rows.append({"Track": "Track C — Typhoon-1B", "หมวดหมู่": pred, "คำอธิบาย": LABEL_DESC.get(pred, ""), "เวลา (ms)": round(ms, 3)})
        else:
            rows.append({"Track": "Track C — Typhoon-1B", "หมวดหมู่": "N/A (โมเดลไม่พร้อม)", "คำอธิบาย": "", "เวลา (ms)": None})

        st.subheader("ผลการทำนายของแต่ละ Track")
        st.table(rows)
