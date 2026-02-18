import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate probe tip keypoints directly on a video with a frame slider."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotation_project"),
        help="Directory where images and COCO JSON will be written.",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="annotations_train.json",
        help="Output COCO JSON filename (inside output-dir/exports).",
    )
    parser.add_argument("--class-name", type=str, default="probe", help="COCO category name.")
    parser.add_argument("--keypoint-name", type=str, default="tip", help="Keypoint name.")
    parser.add_argument(
        "--bbox-size-px",
        type=float,
        default=20.0,
        help="BBox size centered on click point.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to include.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Annotate every Nth frame in slider timeline.",
    )
    parser.add_argument(
        "--resume-json",
        type=Path,
        default=None,
        help="Optional COCO JSON file to preload existing annotations.",
    )
    parser.add_argument(
        "--autosave-on-exit",
        action="store_true",
        help="Write images + JSON automatically when quitting.",
    )
    return parser.parse_args()


def read_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return ok, frame


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_resume_annotations(resume_json: Path) -> dict[int, tuple[float, float]]:
    data = json.loads(resume_json.read_text(encoding="utf-8"))
    image_id_to_file = {
        int(img["id"]): str(img["file_name"]) for img in data.get("images", []) if "id" in img
    }
    result: dict[int, tuple[float, float]] = {}
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        keypoints = ann.get("keypoints", [])
        if image_id is None or len(keypoints) < 2:
            continue
        file_name = image_id_to_file.get(int(image_id))
        if not file_name:
            continue
        stem = Path(file_name).stem
        if not stem.startswith("frame_"):
            continue
        try:
            frame_idx = int(stem.split("_", 1)[1])
        except Exception:
            continue
        result[frame_idx] = (float(keypoints[0]), float(keypoints[1]))
    return result


def save_coco(
    cap: cv2.VideoCapture,
    annotations: dict[int, tuple[float, float]],
    output_dir: Path,
    json_name: str,
    class_name: str,
    keypoint_name: str,
    bbox_size_px: float,
) -> Path:
    images_out = output_dir / "images"
    exports_out = output_dir / "exports"
    images_out.mkdir(parents=True, exist_ok=True)
    exports_out.mkdir(parents=True, exist_ok=True)

    sorted_indices = sorted(annotations.keys())
    images = []
    anns = []
    ann_id = 1

    for image_id, frame_idx in enumerate(sorted_indices, start=1):
        ok, frame = read_frame(cap, frame_idx)
        if not ok or frame is None:
            print(f"[WARN] Could not read frame {frame_idx}, skipping.")
            continue

        h, w = frame.shape[:2]
        x, y = annotations[frame_idx]
        x = clamp(x, 0.0, float(w - 1))
        y = clamp(y, 0.0, float(h - 1))

        file_name = f"frame_{frame_idx:06d}.jpg"
        out_img = images_out / file_name
        if not cv2.imwrite(str(out_img), frame):
            print(f"[WARN] Failed to write image {out_img}, skipping.")
            continue

        half = bbox_size_px / 2.0
        x_min = clamp(x - half, 0.0, float(w - 1))
        y_min = clamp(y - half, 0.0, float(h - 1))
        x_max = clamp(x + half, 0.0, float(w - 1))
        y_max = clamp(y + half, 0.0, float(h - 1))
        bw = max(1.0, x_max - x_min)
        bh = max(1.0, y_max - y_min)

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(w),
                "height": int(h),
            }
        )
        anns.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x_min, y_min, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "num_keypoints": 1,
                "keypoints": [x, y, 2],
            }
        )
        ann_id += 1

    payload = {
        "images": images,
        "annotations": anns,
        "categories": [
            {
                "id": 1,
                "name": class_name,
                "supercategory": "object",
                "keypoints": [keypoint_name],
                "skeleton": [],
            }
        ],
    }

    json_path = exports_out / json_name
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.step < 1:
        raise ValueError("--step must be >= 1.")
    if args.bbox_size_px <= 0:
        raise ValueError("--bbox-size-px must be > 0.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count from video.")

    start = max(0, args.start_frame)
    if start >= frame_count:
        cap.release()
        raise ValueError("--start-frame is beyond video length.")

    frame_indices = list(range(start, frame_count, args.step))
    if not frame_indices:
        cap.release()
        raise RuntimeError("No frames selected by start/step settings.")

    annotations: dict[int, tuple[float, float]] = {}
    if args.resume_json is not None:
        if not args.resume_json.exists():
            cap.release()
            raise FileNotFoundError(f"--resume-json not found: {args.resume_json}")
        annotations = load_resume_annotations(args.resume_json)
        print(f"Loaded {len(annotations)} existing annotations from {args.resume_json}")

    window_name = "Video Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = {"slider_pos": 0}

    def on_slider(pos: int) -> None:
        state["slider_pos"] = int(pos)

    cv2.createTrackbar("frame", window_name, 0, len(frame_indices) - 1, on_slider)

    current_frame_idx = frame_indices[0]
    ok, frame = read_frame(cap, current_frame_idx)
    if not ok or frame is None:
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError(f"Could not read initial frame {current_frame_idx}.")

    def on_mouse(event, x, y, flags, param):
        del flags, param
        idx = state["slider_pos"]
        frame_idx = frame_indices[idx]
        if event == cv2.EVENT_LBUTTONDOWN:
            annotations[frame_idx] = (float(x), float(y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            annotations.pop(frame_idx, None)

    cv2.setMouseCallback(window_name, on_mouse)

    print("Controls:")
    print("  slider: jump through video timeline")
    print("  left click: set keypoint")
    print("  right click: remove keypoint")
    print("  a/d: previous/next slider frame")
    print("  s: save COCO JSON + annotated frame images")
    print("  q: quit")

    while True:
        idx = state["slider_pos"]
        idx = max(0, min(len(frame_indices) - 1, idx))
        frame_idx = frame_indices[idx]

        if frame_idx != current_frame_idx:
            ok, frame = read_frame(cap, frame_idx)
            if not ok or frame is None:
                print(f"[WARN] Could not read frame {frame_idx}")
                state["slider_pos"] = max(0, idx - 1)
                cv2.setTrackbarPos("frame", window_name, state["slider_pos"])
                continue
            current_frame_idx = frame_idx

        disp = frame.copy()
        tip = annotations.get(frame_idx)
        if tip is not None:
            cv2.circle(disp, (int(tip[0]), int(tip[1])), 6, (0, 255, 0), -1)

        text1 = f"frame {frame_idx}/{frame_count - 1} slider {idx + 1}/{len(frame_indices)}"
        text2 = f"annotated={len(annotations)} left=set right=remove s=save q=quit"
        cv2.putText(disp, text1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)
        cv2.putText(disp, text2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 255), 2)
        cv2.imshow(window_name, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            if args.autosave_on_exit and annotations:
                out_json = save_coco(
                    cap=cap,
                    annotations=annotations,
                    output_dir=args.output_dir,
                    json_name=args.json_name,
                    class_name=args.class_name,
                    keypoint_name=args.keypoint_name,
                    bbox_size_px=args.bbox_size_px,
                )
                print(f"Autosaved: {out_json}")
            break
        if key == ord("a"):
            state["slider_pos"] = max(0, idx - 1)
            cv2.setTrackbarPos("frame", window_name, state["slider_pos"])
        elif key == ord("d"):
            state["slider_pos"] = min(len(frame_indices) - 1, idx + 1)
            cv2.setTrackbarPos("frame", window_name, state["slider_pos"])
        elif key == ord("s"):
            out_json = save_coco(
                cap=cap,
                annotations=annotations,
                output_dir=args.output_dir,
                json_name=args.json_name,
                class_name=args.class_name,
                keypoint_name=args.keypoint_name,
                bbox_size_px=args.bbox_size_px,
            )
            print(f"Saved: {out_json}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
