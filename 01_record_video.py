import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record webcam video to an MP4 file.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recordings"),
        help="Directory for output video files.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested FPS.")
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec string (default: mp4v).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"probe_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam. Try --camera with a different index.")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read a frame from webcam.")

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = args.fps if args.fps > 1 else 30.0

    codec = args.codec.strip()
    if len(codec) != 4:
        cap.release()
        raise ValueError("--codec must be exactly 4 characters (FourCC).")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output writer at '{out_path}'. Try a different --codec."
        )

    recording = False
    written_frames = 0

    print(f"Output file: {out_path}")
    print("Controls: r = toggle recording | q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        disp = frame.copy()
        if recording:
            cv2.circle(disp, (20, 20), 8, (0, 0, 255), -1)
            cv2.putText(disp, "REC", (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(frame)
            written_frames += 1

        cv2.imshow("Webcam (recording)", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            recording = not recording
        elif key == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if written_frames == 0:
        print(f"No recorded frames were saved. File path reserved: {out_path}")
    else:
        print(f"Done. Saved {written_frames} frames to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
