import json
import pickle
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT_DIR = Path(__file__).resolve().parent             # 2-train-test-model/llm_prompting/
ROOT = SCRIPT_DIR.parent.parent                          # repo root

# รันแบบ local ด้วย Typhoon (scb10x/typhoon-ai) ผ่าน HuggingFace transformers
MODEL_NAME = "typhoon-ai/llama3.2-typhoon2-1b-instruct"

VALID_LABELS = ["sale_oneoff", "sale_enterprise", "it_support", "admin_general"]

SYSTEM_PROMPT = f"""คุณเป็นระบบจำแนกข้อความแชทลูกค้าของแอป Chat AI ที่รวมหลายโมเดล (GPT, Claude, Gemini ฯลฯ)
เข้า 4 หมวดหมู่เท่านั้น:
- sale_oneoff: ลูกค้ารายบุคคลสนใจซื้อแพ็กเกจครั้งเดียว/ใช้คนเดียว
- sale_enterprise: ลูกค้าองค์กรต้องการสัญญาระยะยาว/หลาย seat/API แบบ custom
- it_support: ปัญหาการใช้งานแอป เช่น error, ล็อกอินไม่ได้, โมเดลตอบช้า
- admin_general: บิล ใบเสร็จ เปลี่ยนข้อมูลบัญชี ยกเลิกสมาชิก

ค่า "label" ต้องเป็นหนึ่งใน {VALID_LABELS} เท่านั้น (คำเดียว ตัวสะกดตรงตามนี้เป๊ะๆ)
ห้ามคัดลอกข้อความของลูกค้ามาใส่ใน "label" เด็ดขาด
ตอบกลับเป็น JSON บรรทัดเดียวเท่านั้น ไม่ต้องอธิบายเพิ่ม รูปแบบ: {{"label": "...", "confidence": 0.0}}"""

FEW_SHOT_EXAMPLES = [
    {"text": "อยากซื้อแพ็กเกจ Pro ครั้งเดียวใช้คนเดียวครับ", "label": "sale_oneoff"},
    {"text": "บริษัทเรามีพนักงาน 50 คน สนใจแพ็กเกจ Enterprise", "label": "sale_enterprise"},
    {"text": "ล็อกอินเข้าแอปไม่ได้ค่ะ ขึ้น error ตลอด", "label": "it_support"},
    {"text": "ขอใบเสร็จเดือนที่แล้วค่ะ", "label": "admin_general"},
]


def build_prompt(text: str) -> str:
    examples_text = "\n".join(
        f'ข้อความ: "{ex["text"]}"\n{{"label": "{ex["label"]}", "confidence": 0.9}}'
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"{examples_text}\n\nข้อความ: \"{text}\""


def parse_label(raw: str) -> dict:
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if parsed.get("label") in VALID_LABELS:
                return parsed
        except json.JSONDecodeError:
            pass
    for label in VALID_LABELS:
        if label in raw:
            return {"label": label, "confidence": None}
    raise ValueError(f"could not parse a valid label from model output: {raw!r}")


def classify_with_llm(text: str, pipe) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(text)},
    ]
    output = pipe(messages, max_new_tokens=30, max_length=None, do_sample=False)
    raw = output[0]["generated_text"][-1]["content"]
    return parse_label(raw)


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[Track C][1/4] โหลดโมเดล {MODEL_NAME} บน {device} ...")
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        dtype=torch.bfloat16,
        device=device,
    )

    print("[Track C][2/4] โหลดข้อมูล mock + แบ่ง train/test ...")
    df = pd.read_csv(ROOT / "1-data" / "mock_conversations_400.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    print(f"  train={len(X_train)} / test={len(X_test)}")

    print(f"[Track C][3/4] classify ทีละข้อความด้วย few-shot prompting ({len(X_test)} ตัวอย่าง) ...")
    
    llm_preds = []
    t0 = time.perf_counter()
    for i, t in enumerate(X_test):
        try:
            result = classify_with_llm(t, pipe)
            llm_preds.append(result["label"])
        except Exception as e:
            print(f"error classifying '{t[:40]}...': {e}")
            llm_preds.append("admin_general")  # fallback label
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(X_test)} classified")
    infer_time_ms_per_sample = (time.perf_counter() - t0) / len(X_test) * 1000

    print("[Track C][4/4] สรุปผล + บันทึก cache ...")
    print("LLM Prompting (Typhoon local):\n", classification_report(y_test, llm_preds))

    with open(SCRIPT_DIR / "preds_cache.pkl", "wb") as f:
        pickle.dump({
            "y_test": list(y_test),
            "llm_preds": llm_preds,
            "infer_time_ms_per_sample": infer_time_ms_per_sample,
        }, f)
    print(f"  บันทึก predictions ไว้ที่ {SCRIPT_DIR / 'preds_cache.pkl'}")
