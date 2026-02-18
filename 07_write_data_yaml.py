import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Ultralytics data YAML for 1-keypoint probe tip pose.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Dataset root containing images/ and labels/.",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=Path("probe_tip.yaml"),
        help="Output YAML path.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="probe",
        help="Single class name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    payload = {
        "path": str(args.dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: args.class_name},
        "kpt_shape": [1, 3],
    }

    args.yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.yaml_path.resolve()}")
    print(args.yaml_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
