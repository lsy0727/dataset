# google open dataset 일부를 다운받음
#!/usr/bin/env python3
# download_dataset.py
# Open Images V6 -> YOLOv5/YOLOv8 포맷 변환
# train 전체 포함, val/test는 only_matching=True
# + 실제 라벨 병합 (--merge), 이름만 변경 (--rename) 지원

import argparse, os
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from fiftyone.utils import openimages as oi
import fiftyone.core.labels as fol  # Detections 타입 확인용

# ---------- Open Images boxable 이름 매핑 (다운로드/필터용) ----------
OI_NAME_MAP = {
    "Shoe": "Footwear",
    "Slipper": "Footwear",
    "Boot": "Footwear",
    "Cup": "Coffee cup",
    "Faucet": "Tap",
    "Sofa": "Couch",
    "Dining table": "Table",
}

# ---------- 실제 라벨 병합 규칙 (원본 → 통합 이름) ----------
MERGE_MAP = {
    # 가방
    "Handbag": "Bag",
    "Backpack": "Bag",
    "Suitcase": "Bag",
    # 테이블 계열
    "Table": "Table",
    "Coffee table": "Table",
    "Desk": "Table",
    "Dining table": "Table",
    "Nightstand": "Table",
    # 착석류
    "Couch": "Chair",
    "Sofa": "Chair",
    "Stool": "Chair",
    "Bench": "Chair",
    # 신발류
    "Footwear": "Shoe",
    "Boot": "Shoe",
    "Slipper": "Shoe",
    # 식기류
    "Wine glass": "Cup",
    "Mug": "Cup",
}

# ---------- 유틸 ----------
def unique(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def map_to_oi_boxable(names):
    """요청 클래스 -> OI V6 boxable 공식명으로 매핑, 없으면 드롭"""
    available = set(oi.get_classes())
    mapped, dropped = [], []
    for n in names:
        m = OI_NAME_MAP.get(n, n)
        if m in available:
            mapped.append(m)
        else:
            dropped.append(n)
    mapped = unique(mapped)
    if dropped:
        print(f"[WARN] Dropped {len(dropped)} non-boxable: {dropped}")
    print(f"[INFO] Using {len(mapped)} boxable classes")
    return mapped

# ---------- 라벨 필드 자동 탐지 ----------
def _has_nonempty_detections(ds: fo.Dataset, field: str) -> bool:
    try:
        v = ds.match(F(field).exists() & (F(f"{field}.detections").length() > 0))
        return len(v) > 0
    except Exception:
        return False

def guess_label_field(ds: fo.Dataset) -> str:
    for cand in ["detections", "ground_truth", "open_images_detections", "objects", "labels"]:
        if _has_nonempty_detections(ds, cand):
            return cand
    try:
        for lf in ds._get_label_fields():
            if _has_nonempty_detections(ds, lf):
                return lf
    except Exception:
        pass
    schema = ds.get_field_schema()
    for name, ftype in schema.items():
        if ftype is fol.Detections or getattr(ftype, "_doc_type", None) is fol.Detections:
            if _has_nonempty_detections(ds, name):
                return name
    raise RuntimeError("라벨 필드를 자동으로 찾지 못했습니다. dataset.get_field_schema()로 확인 필요.")

# ---------- 실제 라벨 병합 ----------
def apply_merge_labels(ds: fo.Dataset, label_field: str, merge_map: dict):
    changed = 0
    it = ds.iter_samples(progress=True)
    for sample in it:
        det = getattr(sample, label_field, None)
        if not isinstance(det, fol.Detections) or not det.detections:
            continue
        touched = False
        for d in det.detections:
            new_label = merge_map.get(d.label, d.label)
            if new_label != d.label:
                d.label = new_label
                touched = True
        if touched:
            sample.set_field(label_field, det)
            sample.save()
            changed += 1
    print(f"[INFO] Merge-applied samples updated: {changed}")

# ---------- 이름만 바꾸는 rename ----------
def parse_rename_arg(s: str):
    if not s:
        return {}
    out = {}
    pairs = [p.strip() for p in s.split(",") if p.strip()]
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--rename 형식 오류: '{p}'. '원래=새이름' 형식으로 적어주세요")
        src, dst = [t.strip() for t in p.split("=", 1)]
        out[src] = dst
    return out

def apply_export_renames(class_list, rename_map):
    out, seen = [], set()
    for name in class_list:
        new_name = rename_map.get(name, name)
        if new_name in seen:
            raise ValueError(f"[ERROR] 이름 충돌: '{new_name}' 가 중복됩니다.")
        seen.add(new_name)
        out.append(new_name)
    return out

# ---------- 클래스 프리셋 ----------
def get_seed_classes():
    return ["Chair", "Shoe", "Bottle", "Cup", "Handbag", "Backpack"]

def get_wide_indoor_classes():
    return [
        "Sofa","Couch","Desk","Table","Coffee table","Dining table",
        "Stool","Bench","Nightstand",
        "Wine glass","Mug",
        "Towel","Toilet paper",
        "Book","Pen","Boot","Slipper",
    ]

# ---------- 메인 ----------
def main():
    parser = argparse.ArgumentParser(description="Open Images V6 indoor-related -> YOLO export")
    parser.add_argument("--out_dir", required=True, help="YOLO export root dir")
    parser.add_argument("--splits", nargs="+", default=["train"], help="train / validation / test (val 별칭 지원)")
    parser.add_argument("--include_seed_classes", action="store_true")
    parser.add_argument("--include_wide_indoor", action="store_true")
    parser.add_argument("--max_per_class", type=int, default=0)
    parser.add_argument("--yolo_split_ratio", type=float, default=0.9)
    parser.add_argument("--merge", action="store_true", help="MERGE_MAP에 따라 실제 라벨 통합 처리 후 내보냄")
    parser.add_argument("--rename", type=str, default="", help='클래스 이름만 바꾸기 (예: "Microwave oven=Oven")')
    args = parser.parse_args()

    # split 정규화
    _alias = {"val": "validation", "valid": "validation"}
    args.splits = [_alias.get(s.lower(), s.lower()) for s in args.splits]

    # 클래스 구성
    req = []
    if args.include_seed_classes:
        req += get_seed_classes()
    if args.include_wide_indoor:
        req += get_wide_indoor_classes()
    req = unique(req)
    if not req:
        print("No classes specified"); return
    print(f"[INFO] Requested {len(req)} classes: {req}")
    classes = map_to_oi_boxable(req)

    # 병합용 데이터셋
    merged_name = "oi_indoor_merge"
    if merged_name in fo.list_datasets():
        fo.delete_dataset(merged_name)
    merged_ds = fo.Dataset(merged_name)

    # split별 다운로드/병합
    split_datasets = {}
    for split in args.splits:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split '{split}'.")
        only_match = (split != "train")  # ✅ train만 전체 포함
        print(f"[INFO] Loading Open Images V6 split={split} (only_matching={only_match}) ...")
        ds = foz.load_zoo_dataset(
            "open-images-v6",
            split=split,
            label_types=["detections"],
            classes=classes,
            only_matching=only_match,
        )
        split_datasets[split] = ds
        merged_ds.merge_samples(ds)
        print(f"[INFO] Merged size: {len(merged_ds)}")

    # 라벨 필드
    label_field = guess_label_field(merged_ds)
    print("[INFO] Using label field:", label_field)

    # --merge 적용
    if args.merge:
        print("[INFO] Applying label merge to each split...")
        for split, ds in split_datasets.items():
            print(f"  - split={split}")
            apply_merge_labels(ds, label_field, MERGE_MAP)

    # 빈 라벨 제거
    merged_ds = merged_ds.match(F(label_field).exists() & (F(f"{label_field}.detections").length() > 0))
    print("[INFO] After filtering (merged):", len(merged_ds))
    if len(merged_ds) == 0:
        print("[WARN] No samples to export."); return

    # 최종 클래스 리스트
    if args.merge:
        classes_out = []
        seen = set()
        for c in classes:
            mc = MERGE_MAP.get(c, c)
            if mc not in seen:
                classes_out.append(mc); seen.add(mc)
    else:
        classes_out = list(classes)

    rename_map = parse_rename_arg(args.rename)
    classes_out = apply_export_renames(classes_out, rename_map)

    print("[INFO] YOLO class index mapping:")
    for i, name in enumerate(classes_out):
        print(f"  {i}: {name}")

    # export
    export_root = os.path.abspath(args.out_dir)
    os.makedirs(export_root, exist_ok=True)

    for split, view in split_datasets.items():
        view = view.match(F(label_field).exists() & (F(f"{label_field}.detections").length() > 0))
        if len(view) == 0:
            print(f"[WARN] No samples in split={split} after filtering; skipping")
            continue
        yolo_split = "val" if split == "validation" else split
        print(f"[INFO] Exporting split={yolo_split} → {export_root}")
        view.export(
            export_dir=export_root,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes_out,
            split=yolo_split,
        )

    print("[✅ DONE] Export complete at", export_root)
    print("Structure: images/{train,val,test}  labels/{train,val,test}  + classes.txt, data.yaml")

if __name__ == "__main__":
    main()
