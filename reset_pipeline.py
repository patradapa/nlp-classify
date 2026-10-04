"""
ลบทุกอย่างที่เกิดจากการ "รัน" pipeline นี้ ให้กลับไปเหมือนยังไม่เคยรันอะไรเลย
(model checkpoint, prediction cache, ผล evaluate, __pycache__)

ของที่ "ไม่ลบ": mock data CSV (1-data/mock_conversations_400.csv), venv/, requirements.txt,
source code ทั้งหมด (.py), README, สเปกต้นฉบับ
และ HuggingFace cache ของเครื่อง (~/.cache/huggingface) — ไม่ได้อยู่ใน repo นี้ ไม่แตะ

ใช้งาน:
    python reset_pipeline.py            # ถามยืนยันก่อนลบจริง
    python reset_pipeline.py --dry-run  # แสดงว่าจะลบอะไร ไม่ลบจริง
    python reset_pipeline.py -y         # ลบจริงทันทีไม่ถามยืนยัน

รันจากที่ไหนก็ได้ — path anchor ด้วยตำแหน่งไฟล์นี้เอง
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (path สัมพัทธ์จาก ROOT, คำอธิบาย)
TARGETS = [
    ("2-train-test-model/wangchanberta/wangchanberta-chat-classifier", "WangchanBERTa checkpoint (Track B)"),
    ("2-train-test-model/wangchanberta/preds_cache.pkl", "Track B cached predictions"),
    ("2-train-test-model/llm_prompting/preds_cache.pkl", "Track C cached predictions"),
    ("3-evaluate/results", "ผลรวมจาก evaluate_all.py ทั้งหมด"),
]

# โฟลเดอร์ของ pipeline เอง ที่ควรเดินหา __pycache__/runs/wandb/mlruns ข้างใน
# (ไม่เดินเข้า .git หรือ venv เพราะไม่เกี่ยวและเปลืองเวลา)
PROJECT_DIRS = ["1-data", "2-train-test-model", "3-evaluate"]
STRAY_DIR_NAMES = {"__pycache__", "runs", "wandb", "mlruns"}


def find_stray_dirs():
    found = []
    for project_dir in PROJECT_DIRS:
        base = ROOT / project_dir
        if not base.exists():
            continue
        for name in STRAY_DIR_NAMES:
            found.extend(sorted(base.rglob(name)))
    return found


def human_size(path: Path) -> str:
    if path.is_file():
        total = path.stat().st_size
    else:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.0f}{unit}"
        total /= 1024
    return f"{total:.1f}TB"


def remove(path: Path, dry_run: bool):
    if path.is_dir():
        if not dry_run:
            shutil.rmtree(path)
    else:
        if not dry_run:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="แสดงรายการที่จะลบ ไม่ลบจริง")
    parser.add_argument("-y", "--yes", action="store_true", help="ลบจริงทันทีไม่ถามยืนยัน")
    args = parser.parse_args()

    to_delete = []
    for rel_path, desc in TARGETS:
        path = ROOT / rel_path
        if path.exists():
            to_delete.append((path, desc))

    stray_dirs = [p for p in find_stray_dirs() if p.exists()]
    for p in stray_dirs:
        to_delete.append((p, f"{p.name}/ (artifact ที่ pipeline/dependency สร้างไว้)"))

    if not to_delete:
        print("ไม่มีอะไรให้ลบ — repo สะอาดอยู่แล้ว (เหมือนยังไม่เคยรัน)")
        return

    print(("จะลบ" if not args.dry_run else "[dry-run] จะลบ (ถ้าไม่ใส่ --dry-run)") + ":")
    for path, desc in to_delete:
        rel = path.relative_to(ROOT)
        print(f"  - {rel}  [{human_size(path)}]  — {desc}")

    if args.dry_run:
        print("\n(dry-run — ยังไม่ลบอะไรจริง)")
        return

    if not args.yes:
        confirm = input(f"\nยืนยันลบ {len(to_delete)} รายการข้างบนจริงหรือไม่? พิมพ์ 'yes' เพื่อลบ: ")
        if confirm.strip().lower() != "yes":
            print("ยกเลิก — ไม่ได้ลบอะไร")
            return

    for path, desc in to_delete:
        remove(path, dry_run=False)
        print(f"  ลบแล้ว: {path.relative_to(ROOT)}")

    print(
        "\nเสร็จแล้ว — repo กลับไปเหมือนยังไม่เคยรัน pipeline นี้เลย (เว้น mock data CSV ที่เก็บไว้)\n"
        "รันตามลำดับใน README.md หัวข้อ 5 เพื่อเริ่มใหม่ตั้งแต่ต้น"
    )


if __name__ == "__main__":
    main()
