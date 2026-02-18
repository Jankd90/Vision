import argparse
import platform
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check environment dependencies for the Vision pipeline."
    )
    parser.add_argument(
        "--check-camera",
        action="store_true",
        help="Attempt to open a webcam and read one frame.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index used with --check-camera (default: 0).",
    )
    return parser.parse_args()


def import_checks() -> tuple[bool, list[str]]:
    missing = []
    for module_name in ("cv2", "numpy", "ultralytics", "yaml"):
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)
    return (len(missing) == 0, missing)


def check_camera(camera_index: int) -> bool:
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}.")
        return False

    ok, _ = cap.read()
    cap.release()
    if not ok:
        print(f"[ERROR] Camera index {camera_index} opened but no frame was read.")
        return False
    print(f"[OK] Camera index {camera_index} opened and frame read successfully.")
    return True


def main() -> int:
    args = parse_args()
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")

    ok, missing = import_checks()
    if not ok:
        print("[ERROR] Missing required modules:")
        for name in missing:
            print(f"  - {name}")
        print("Install dependencies with:")
        print("  pip install -r requirements.txt")
        return 1

    print("[OK] All required Python modules are importable.")

    if args.check_camera:
        if not check_camera(args.camera):
            return 2

    print("Environment check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
