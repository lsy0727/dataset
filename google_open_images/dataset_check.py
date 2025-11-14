# 데이터셋의 라벨링된 객체 수를 파악하기 위한 스크립트
#!/usr/bin/env python3
# count_dataset_stats.py
# Usage: python count_dataset_stats.py --root dataset/open_images_dataset

import argparse
from pathlib import Path
from collections import defaultdict
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_classes(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        # {id: name} 형태도 지원
        max_id = max(int(k) for k in names.keys())
        classes = [None]*(max_id+1)
        for k,v in names.items():
            classes[int(k)] = v
    elif isinstance(names, list):
        classes = names
    else:
        raise ValueError("data.yaml에서 'names'를 찾을 수 없습니다.")
    return classes

def list_images(split_images_dir: Path):
    return [p for p in split_images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

def parse_label_file(txt_path: Path):
    """
    YOLO 형식 가정: 각 줄의 첫 토큰이 class_id (정수).
    이후 좌표/세그/키포인트 등은 무시. 빈 줄/주석 무시.
    """
    class_ids = []
    if not txt_path.exists():
        return class_ids
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                cid = int(float(parts[0]))  # '0', '0.0' 모두 허용
                class_ids.append(cid)
            except Exception:
                # 잘못된 라벨 라인은 스킵
                continue
    return class_ids

def count_split(root: Path, split: str, classes: list[str]):
    """
    반환:
      total_images: 이미지 총 수
      images_with_class: {class_id: 해당 클래스를 포함한 이미지 수}
      instances: {class_id: 총 인스턴스 수}
    """
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split

    image_files = list_images(images_dir)
    total_images = len(image_files)

    images_with_class = defaultdict(int)
    instances = defaultdict(int)

    # 라벨 파일을 기준으로 순회 (라벨 없는 이미지는 클래스 카운트에 영향 없음)
    for lbl_path in labels_dir.rglob("*.txt"):
        class_ids = parse_label_file(lbl_path)
        if not class_ids:
            continue
        # 해당 파일 내 클래스별 등장 횟수 집계
        per_image_counts = defaultdict(int)
        for cid in class_ids:
            if cid < 0 or cid >= len(classes):
                # data.yaml의 클래스 범위를 벗어나면 스킵
                continue
            per_image_counts[cid] += 1

        # 이 이미지에서 등장한 클래스마다:
        #   - images_with_class 는 1씩(이미지 존재 카운트)
        #   - instances 는 등장 횟수만큼
        for cid, cnt in per_image_counts.items():
            images_with_class[cid] += 1
            instances[cid] += cnt

    return total_images, images_with_class, instances

def print_report(split: str, total_images: int, images_with_class: dict, instances: dict, classes: list[str]):
    print(f"\n=== [{split.upper()}] ===")
    print(f"총 이미지 수: {total_images}")
    print(f"{'class_id':>8}  {'class_name':<30}  {'images_with_class':>16}  {'instances':>10}")
    print("-"*72)
    for cid, name in enumerate(classes):
        img_cnt = images_with_class.get(cid, 0)
        inst_cnt = instances.get(cid, 0)
        print(f"{cid:8d}  {name:<30}  {img_cnt:16d}  {inst_cnt:10d}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/linux/yolov8_ws/dataset/open_images_dataset",
                    help="데이터셋 루트 경로 (images/, labels/, data.yaml 포함)")
    args = ap.parse_args()

    root = Path(args.root)
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {yaml_path}")

    classes = load_classes(yaml_path)

    # train
    tr_total, tr_img_with, tr_inst = count_split(root, "train", classes)
    # val
    va_total, va_img_with, va_inst = count_split(root, "val", classes)

    # 출력
    print_report("train", tr_total, tr_img_with, tr_inst, classes)
    print_report("val", va_total, va_img_with, va_inst, classes)

if __name__ == "__main__":
    main()
