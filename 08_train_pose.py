import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO pose model for probe tip keypoint tracking.")
    parser.add_argument("--data-yaml", type=Path, default=Path("probe_tip.yaml"), help="Path to data YAML.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-pose.pt",
        help="Base Ultralytics pose model checkpoint.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--name", type=str, default="probe_tip_pose", help="Run name under runs/pose.")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='Torch device (examples: "cpu", "0"). Empty means Ultralytics auto-select.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data_yaml}")

    model = YOLO(args.model)
    kwargs = dict(
        data=str(args.data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
    )
    if args.device.strip():
        kwargs["device"] = args.device

    model.train(**kwargs)
    print("Training finished.")
    print(f"Expected best weights at: runs/pose/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
