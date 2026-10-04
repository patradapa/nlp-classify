import csv
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).resolve().parent / "mock_conversations_400.csv"

GREETINGS = [
    "สวัสดีค่ะ", "สวัสดีครับ", "รบกวนสอบถามหน่อยค่ะ", "หวัดดีครับ",
    "ขอโทษที่รบกวนค่ะ", "เรียนทีมงานค่ะ", "สอบถามหน่อยนะคะ", "ขอถามอะไรหน่อยครับ",
    "แอดมินคะ", "รบกวนหน่อยนะครับ",
]

PLANS = ["Starter", "Basic", "Pro", "Individual", "Lite", "Personal"]
SEAT_COUNTS = [5, 8, 10, 12, 15, 20, 25, 30, 50, 100]
CONTRACT_YEARS = [1, 2, 3]
MODELS = ["GPT", "Claude", "Gemini", "Llama", "Mistral"]

# emoji จงใจแจกแบบสุ่มเท่ากันทุก label เพื่อไม่ให้เป็น shortcut feature (data leakage)
EMOJIS = ["😊", "🙏", "😅", "😢", "😡", "👍", "🤔", "😭", "‼️", "❓"]
EMOJI_PROB = 0.25

# คำลงท้ายแบบไม่เป็นทางการ/ยืดเสียง จงใจแจกแบบสุ่มเท่ากันทุก label เช่นกัน
CASUAL_ENDINGS = {
    "ครับ": ["ครับบบบ", "ค้าบบบบ", "คร่าบ"],
    "ค่ะ": ["ค่ะะะะ", "ค่าาา", "ค๊า"],
    "คะ": ["คะะะะ", "ค๊ะ"],
}
CASUAL_PROB = 0.25


def casualize(text):
    for formal, variants in CASUAL_ENDINGS.items():
        if text.endswith(formal):
            return text[: -len(formal)] + random.choice(variants)
    return text


# บางส่วนทำเป็นบทสนทนาหลายข้อความ (ลูกค้าพิมพ์แยกหลายบับเบิล + แอดมินรับเรื่อง) แทนประโยคเดียว
ADMIN_ACKS = [
    "ได้ค่ะ รอสักครู่นะคะ",
    "เดี๋ยวแอดมินเช็คให้ครับ",
    "รับทราบค่ะ ขอข้อมูลเพิ่มอีกนิดนะคะ",
    "โอเคครับ รอแป๊บนะครับ",
]
MULTI_TURN_PROB = 0.2
ADMIN_ACK_PROB = 0.5
FOLLOWUP_PROB = 0.5


def build_conversation(greeting, body, bodies_pool, fill_fn):
    turns = [f"ลูกค้า: {greeting or random.choice(GREETINGS)}"]
    if random.random() < ADMIN_ACK_PROB:
        turns.append(f"แอดมิน: {random.choice(ADMIN_ACKS)}")
    turns.append(f"ลูกค้า: {body}")
    if random.random() < FOLLOWUP_PROB:
        followup = fill_fn(random.choice(bodies_pool))
        turns.append(f"ลูกค้า: {followup}")
    return "\n".join(turns)

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
    # ประโยคกำกวม ใช้คำที่คาบเกี่ยวกับ sale_enterprise (ทีม/สัญญา/API) แต่เนื้อหายังเป็นการซื้อรายบุคคล
    "ทีมเล็กๆ ของเรา 3-4 คน อยากซื้อแพ็กเกจ {plan} แบบรายคนค่ะ",
    "อยากทำสัญญาแพ็กเกจ {plan} ระยะยาว 1 ปี ล็อกราคาได้ไหมคะ",
    "แพ็กเกจ {plan} มี API ให้เรียกใช้ไหมคะ อยากต่อกับระบบของตัวเอง",
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
    # ประโยคกำกวม ใช้คำที่คาบเกี่ยวกับ sale_oneoff (ทีมเล็ก/ทดลองฟรี/ถามราคา) แต่ยังเป็นความต้องการระดับองค์กร
    "บริษัทเราเล็กๆ มีแค่ {n} คน แต่อยากได้สิทธิ์แบบ Enterprise ได้ไหมคะ",
    "สนใจทดลองใช้ฟรีก่อนตัดสินใจเซ็นสัญญา Enterprise ได้ไหมครับ",
    "อยากทราบราคาแพ็กเกจ Enterprise สำหรับทีมเล็กๆ {n} คนค่ะ",
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
    # ประโยคกำกวม ใช้คำที่คาบเกี่ยวกับ admin_general (จ่ายเงิน/เปลี่ยนอีเมล/ยกเลิกสมาชิก) แต่ต้นเหตุคือบั๊กของระบบ
    "จ่ายเงินไปแล้วแต่ระบบยังบอกว่ายังไม่อัปเกรดแพ็กเกจเลยค่ะ",
    "เปลี่ยนอีเมลในระบบแล้วแต่ login เข้าไม่ได้เลยครับ ขึ้น error",
    "กดยกเลิกสมาชิกแล้วแอปค้าง ไม่ยืนยันอะไรเลยค่ะ",
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
    # ประโยคกำกวม ใช้คำที่คาบเกี่ยวกับ it_support (error/อัปโหลด/แจ้งเตือนระบบ) แต่สิ่งที่ต้องการคือจัดการบัญชี/บิล
    "ระบบขึ้น error ตอนจะเปลี่ยนวิธีชำระเงิน รบกวนช่วยแก้ให้หน่อยค่ะ",
    "อัปโหลดเอกสารยืนยันตัวตนไม่ผ่าน ขอให้ช่วยเปลี่ยนชื่อบัญชีให้หน่อยครับ",
    "แอปแจ้งว่าบัญชีถูกระงับ อยากทราบว่าต้องทำยังไงถึงจะเปิดใช้งานได้อีกครั้งคะ",
]


def gen_rows(bodies, label, count, fill_fn):
    rows = set()
    attempts = 0
    while len(rows) < count and attempts < count * 30:
        attempts += 1
        greeting = random.choice(GREETINGS) if random.random() > 0.15 else ""
        body_template = random.choice(bodies)
        body = fill_fn(body_template)
        if random.random() < MULTI_TURN_PROB:
            text = build_conversation(greeting, body, bodies, fill_fn)
        else:
            text = f"{greeting} {body}".strip()
        if random.random() < CASUAL_PROB:
            text = casualize(text)
        if random.random() < EMOJI_PROB:
            text = f"{text} {random.choice(EMOJIS)}"
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

# label noise จำลองความผิดพลาดของ human annotator จริง (~ไม่กี่% ของข้อมูล)
# ให้สลับไปเฉพาะหมวดที่คาบเกี่ยวกัน (sale_oneoff<->sale_enterprise, it_support<->admin_general)
# เพื่อให้ realistic เหมือน annotator สับสนระหว่าง 2 หมวดที่ใกล้เคียงกัน ไม่ใช่สุ่มมั่ว
LABEL_NOISE_PROB = 0.06
CONFUSABLE_NEIGHBOR = {
    "sale_oneoff": "sale_enterprise",
    "sale_enterprise": "sale_oneoff",
    "it_support": "admin_general",
    "admin_general": "it_support",
}
all_rows = [
    (text, CONFUSABLE_NEIGHBOR[label] if random.random() < LABEL_NOISE_PROB else label)
    for text, label in all_rows
]

random.shuffle(all_rows)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text", "label"])
    for i, (text, label) in enumerate(all_rows, start=1):
        writer.writerow([i, text, label])

from collections import Counter
print(Counter(label for _, label in all_rows))
print("total:", len(all_rows))
