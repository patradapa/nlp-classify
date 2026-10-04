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
