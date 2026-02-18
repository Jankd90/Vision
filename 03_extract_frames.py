import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video for annotation.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frames"),
        help="Parent output directory (subfolder by video stem is created).",
    )
    parser.add_argument("--step", type=int, default=5, help="Save every Nth frame.")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames (1-100).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step < 1:
        raise ValueError("--step must be >= 1.")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be between 1 and 100.")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    frames_dir = args.output_dir / args.video.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    i = 0
    saved = 0
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.step == 0:
            frame_path = frames_dir / f"{i:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame, params):
                raise RuntimeError(f"Failed to write frame: {frame_path}")
            saved += 1
        i += 1

    cap.release()
    print(f"Read {i} frames total.")
    print(f"Saved {saved} frames to: {frames_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
