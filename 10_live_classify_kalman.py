import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from kalman import TipKalman2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam inference for probe tip detection with Kalman filtering."
    )
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
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--process-var",
        type=float,
        default=2.0,
        help="Kalman process noise scalar.",
    )
    parser.add_argument(
        "--meas-var",
        type=float,
        default=36.0,
        help="Kalman measurement noise scalar.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Override FPS for Kalman dt. Use <=0 to auto-estimate from camera.",
    )
    return parser.parse_args()


def _best_index(res) -> int | None:
    if res.keypoints is None or len(res.keypoints) == 0:
        return None
    if res.boxes is None or len(res.boxes) == 0:
        return 0
    confs = res.boxes.conf.cpu().numpy()
    return int(confs.argmax())


def main() -> int:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not (0.0 <= args.conf_thres <= 1.0):
        raise ValueError("--conf-thres must be in [0,1].")
    if args.process_var <= 0 or args.meas_var <= 0:
        raise ValueError("--process-var and --meas-var must be > 0.")

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for inference.")

    fps = args.fps
    if fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 5:
        fps = 30.0
    dt = 1.0 / fps

    kf = TipKalman2D(dt=dt, process_var=args.process_var, meas_var=args.meas_var)
    print(f"Kalman dt={dt:.4f} sec (fps={fps:.2f})")
    print("Controls: q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model(frame, verbose=False, conf=args.conf_thres)[0]
        idx = _best_index(res)
        tip_xy = None

        if idx is not None:
            k = res.keypoints.xy[idx][0].cpu().numpy()
            tip_xy = (float(k[0]), float(k[1]))

        if tip_xy is not None and not kf.initialized:
            kf.init(tip_xy[0], tip_xy[1])

        filt_xy = None
        if kf.initialized:
            filt_xy = kf.predict()
            if tip_xy is not None:
                filt_xy = kf.update(tip_xy[0], tip_xy[1])

        if tip_xy is not None:
            cv2.circle(frame, (int(tip_xy[0]), int(tip_xy[1])), 6, (0, 255, 0), -1)

        if filt_xy is not None:
            cv2.circle(frame, (int(filt_xy[0]), int(filt_xy[1])), 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                "KF",
                (int(filt_xy[0]) + 10, int(filt_xy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        cv2.imshow("YOLO Pose Tip + Kalman", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
