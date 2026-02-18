import argparse
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a manual annotation handoff workspace and instructions."
    )
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory of extracted frames.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("annotation_project"),
        help="Workspace directory for annotation instructions and exports.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="probe",
        help="Single class name for object instances.",
    )
    parser.add_argument(
        "--keypoint-name",
        type=str,
        default="tip",
        help="Single keypoint name for pose annotation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {args.frames_dir}")

    frame_files = sorted(
        p for p in args.frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not frame_files:
        raise RuntimeError(f"No image files found in: {args.frames_dir}")

    args.project_dir.mkdir(parents=True, exist_ok=True)
    exports_dir = args.project_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.project_dir / "frames_manifest.txt"
    manifest_path.write_text(
        "\n".join(str(p.resolve()) for p in frame_files) + "\n", encoding="utf-8"
    )

    instructions_path = args.project_dir / "README_ANNOTATION.txt"
    instructions = f"""Annotation Handoff
==================

Input frames directory:
{args.frames_dir.resolve()}

Frame manifest:
{manifest_path.resolve()}

Labeling spec:
- One class: {args.class_name}
- One keypoint: {args.keypoint_name}
- For each instance, label one keypoint and a small bbox around the tip.
- Suggested bbox size: ~20x20 px around the tip.
- Visibility values: use 2 for visible points by default.

Export requirements:
- COCO Keypoints JSON format.
- Place exported JSON files in:
  {exports_dir.resolve()}
- Expected filenames:
  annotations_train.json
  annotations_val.json

Next pipeline step:
Run 05_convert_coco_to_yolo_pose.py for each exported JSON.
"""
    instructions_path.write_text(instructions, encoding="utf-8")

    print(f"Found {len(frame_files)} frame images.")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote instructions: {instructions_path}")
    print(f"Export target directory: {exports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
