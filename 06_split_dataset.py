import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split labeled images into Ultralytics train/val dataset folders."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing source images.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Directory containing YOLO .txt labels.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Output dataset root directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio in (0,1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="Copy files into dataset (default behavior).")
    mode.add_argument("--move", action="store_true", help="Move files into dataset.")
    return parser.parse_args()


def _split_count(total: int, val_ratio: float) -> tuple[int, int]:
    val_count = max(1, int(round(total * val_ratio)))
    val_count = min(val_count, total - 1)
    train_count = total - val_count
    return train_count, val_count


def _ensure_dirs(dataset_dir: Path) -> dict[str, Path]:
    dirs = {
        "train_images": dataset_dir / "images" / "train",
        "val_images": dataset_dir / "images" / "val",
        "train_labels": dataset_dir / "labels" / "train",
        "val_labels": dataset_dir / "labels" / "val",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _copy_or_move(src: Path, dst: Path, do_move: bool) -> None:
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0,1).")

    image_paths = sorted(
        p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    image_by_stem = {p.stem: p for p in image_paths}

    label_paths = sorted(p for p in args.labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    label_by_stem = {p.stem: p for p in label_paths}

    orphan_labels = sorted(stem for stem in label_by_stem if stem not in image_by_stem)
    unlabeled_images = sorted(stem for stem in image_by_stem if stem not in label_by_stem)

    paired_stems = sorted(set(image_by_stem).intersection(label_by_stem))
    if len(paired_stems) < 2:
        raise RuntimeError(
            "Need at least 2 matched image/label pairs to split train/val. "
            f"Found {len(paired_stems)}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(paired_stems)
    train_count, val_count = _split_count(len(paired_stems), args.val_ratio)
    train_stems = paired_stems[:train_count]
    val_stems = paired_stems[train_count : train_count + val_count]

    dirs = _ensure_dirs(args.dataset_dir)
    do_move = args.move

    for stem in train_stems:
        _copy_or_move(image_by_stem[stem], dirs["train_images"] / image_by_stem[stem].name, do_move)
        _copy_or_move(label_by_stem[stem], dirs["train_labels"] / label_by_stem[stem].name, do_move)

    for stem in val_stems:
        _copy_or_move(image_by_stem[stem], dirs["val_images"] / image_by_stem[stem].name, do_move)
        _copy_or_move(label_by_stem[stem], dirs["val_labels"] / label_by_stem[stem].name, do_move)

    print(f"Dataset root: {args.dataset_dir.resolve()}")
    print(f"Matched pairs: {len(paired_stems)}")
    print(f"Train pairs: {len(train_stems)}")
    print(f"Val pairs: {len(val_stems)}")
    print(f"Orphan labels (no image match): {len(orphan_labels)}")
    print(f"Unlabeled images (no label match): {len(unlabeled_images)}")
    if orphan_labels:
        print("Sample orphan labels:", ", ".join(orphan_labels[:5]))
    if unlabeled_images:
        print("Sample unlabeled images:", ", ".join(unlabeled_images[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
