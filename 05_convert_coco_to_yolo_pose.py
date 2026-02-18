import argparse
import json
import math
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO keypoints JSON to Ultralytics YOLO pose label files."
    )
    parser.add_argument("--coco-json", type=Path, required=True, help="Path to COCO keypoints JSON.")
    parser.add_argument("--labels-out", type=Path, required=True, help="Output directory for YOLO .txt labels.")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Optional image root used to infer missing image width/height.",
    )
    parser.add_argument("--class-id", type=int, default=0, help="Class ID for all labels.")
    parser.add_argument(
        "--keypoint-index",
        type=int,
        default=0,
        help="Keypoint index in COCO keypoints list triplets (x,y,v).",
    )
    parser.add_argument(
        "--bbox-size-px",
        type=float,
        default=20.0,
        help="Fallback bbox size in pixels when COCO bbox is missing.",
    )
    return parser.parse_args()


def _image_size_from_file(images_root: Path, file_name: str) -> tuple[float, float] | tuple[None, None]:
    img_path = images_root / file_name
    if not img_path.exists():
        return None, None
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return float(w), float(h)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> int:
    args = parse_args()
    if not args.coco_json.exists():
        raise FileNotFoundError(f"COCO JSON not found: {args.coco_json}")
    if args.class_id < 0:
        raise ValueError("--class-id must be >= 0.")
    if args.keypoint_index < 0:
        raise ValueError("--keypoint-index must be >= 0.")
    if args.bbox_size_px <= 0:
        raise ValueError("--bbox-size-px must be > 0.")

    args.labels_out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.coco_json.read_text(encoding="utf-8"))

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    img_by_id = {img.get("id"): img for img in images if "id" in img}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        anns_by_img.setdefault(img_id, []).append(ann)

    stats = {
        "images_in_json": len(images),
        "annotations_in_json": len(annotations),
        "images_with_annotations": len(anns_by_img),
        "missing_image_entries": 0,
        "missing_dimensions": 0,
        "malformed_keypoints": 0,
        "written_label_files": 0,
        "written_label_lines": 0,
        "empty_images_skipped": 0,
    }

    for image_id, anns in anns_by_img.items():
        img_meta = img_by_id.get(image_id)
        if img_meta is None:
            stats["missing_image_entries"] += 1
            continue

        file_name = img_meta.get("file_name")
        if not file_name:
            stats["missing_image_entries"] += 1
            continue

        w = img_meta.get("width")
        h = img_meta.get("height")
        if not isinstance(w, (int, float)) or not isinstance(h, (int, float)) or w <= 1 or h <= 1:
            if args.images_root is not None:
                fw, fh = _image_size_from_file(args.images_root, file_name)
                if fw is not None and fh is not None:
                    w, h = fw, fh
            if not isinstance(w, (int, float)) or not isinstance(h, (int, float)) or w <= 1 or h <= 1:
                stats["missing_dimensions"] += 1
                continue

        w = float(w)
        h = float(h)
        lines: list[str] = []

        for ann in anns:
            kpts = ann.get("keypoints", [])
            need = 3 * (args.keypoint_index + 1)
            if not isinstance(kpts, list) or len(kpts) < need:
                stats["malformed_keypoints"] += 1
                continue

            x_px = float(kpts[3 * args.keypoint_index + 0])
            y_px = float(kpts[3 * args.keypoint_index + 1])
            v = int(kpts[3 * args.keypoint_index + 2])
            if not (math.isfinite(x_px) and math.isfinite(y_px)):
                stats["malformed_keypoints"] += 1
                continue

            bbox = ann.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                x_min, y_min, bw, bh = map(float, bbox)
                if bw <= 0 or bh <= 0:
                    bbox = None
            else:
                bbox = None

            if bbox is not None:
                x_min, y_min, bw, bh = map(float, bbox)
                x_c = (x_min + bw / 2.0) / w
                y_c = (y_min + bh / 2.0) / h
                bw_n = bw / w
                bh_n = bh / h
            else:
                half = args.bbox_size_px / 2.0
                x_min = max(0.0, x_px - half)
                y_min = max(0.0, y_px - half)
                x_max = min(w - 1.0, x_px + half)
                y_max = min(h - 1.0, y_px + half)
                bw = max(1.0, x_max - x_min)
                bh = max(1.0, y_max - y_min)
                x_c = (x_min + bw / 2.0) / w
                y_c = (y_min + bh / 2.0) / h
                bw_n = bw / w
                bh_n = bh / h

            kx = x_px / w
            ky = y_px / h
            lines.append(
                f"{args.class_id} "
                f"{_clamp01(x_c):.6f} {_clamp01(y_c):.6f} {_clamp01(bw_n):.6f} {_clamp01(bh_n):.6f} "
                f"{_clamp01(kx):.6f} {_clamp01(ky):.6f} {v}"
            )

        if not lines:
            stats["empty_images_skipped"] += 1
            continue

        label_path = args.labels_out / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["written_label_files"] += 1
        stats["written_label_lines"] += len(lines)

    print(f"Converted labels written to: {args.labels_out.resolve()}")
    print("Conversion stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
