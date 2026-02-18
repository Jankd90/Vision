import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam inference for probe tip detection (no filter).")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/pose/probe_tip_pose/weights/best.pt"),
        help="Path to trained YOLO pose weights.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold for rendering detections.",
    )
    return parser.parse_args()


def _best_index(res) -> tuple[int | None, float | None]:
    if res.keypoints is None or len(res.keypoints) == 0:
        return None, None
    if res.boxes is None or len(res.boxes) == 0:
        return 0, None
    confs = res.boxes.conf.cpu().numpy()
    idx = int(confs.argmax())
    return idx, float(confs[idx])


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not (0.0 <= args.conf_thres <= 1.0):
        raise ValueError("--conf-thres must be in [0,1].")

    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for inference.")

    print("Controls: q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model(frame, verbose=False, conf=args.conf_thres)[0]
        idx, conf = _best_index(res)
        tip_xy = None

        if idx is not None:
            k = res.keypoints.xy[idx][0].cpu().numpy()
            tip_xy = (int(k[0]), int(k[1]))

        if tip_xy is not None:
            cv2.circle(frame, tip_xy, 6, (0, 255, 0), -1)
            if conf is not None:
                cv2.putText(
                    frame,
                    f"conf={conf:.2f}",
                    (tip_xy[0] + 10, tip_xy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("YOLO Pose Tip (no filter)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
