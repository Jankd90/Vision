import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review a recorded video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    print("Controls: space = pause/resume | q = quit")
    paused = False
    frame = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
        if frame is None:
            break
        cv2.imshow("Playback", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(" "):
            paused = not paused
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
