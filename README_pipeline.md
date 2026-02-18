# Vision Probe Tip Pipeline

This repository now includes a sequential CLI pipeline extracted from `probe_tip_tracking_lecture.ipynb`.

## 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
python 00_setup_env.py --check-camera --camera 0
```

## 2. Record a video

```powershell
python 01_record_video.py --camera 0 --output-dir recordings --width 1280 --height 720 --fps 30 --codec mp4v
```

Controls:
- `r`: start/stop recording
- `q`: quit

## 3. Review the recording

```powershell
python 02_review_video.py --video recordings\probe_YYYYMMDD_HHMMSS.mp4
```

Controls:
- `Space`: pause/resume
- `q`: quit

## 4. Extract frames for annotation

```powershell
python 03_extract_frames.py --video recordings\probe_YYYYMMDD_HHMMSS.mp4 --output-dir frames --step 5 --jpeg-quality 95
```

This creates: `frames\<video_stem>\*.jpg`.

## 5. Annotation options

### Option A: Built-in video annotator with slider (recommended)

```powershell
python annotate.py --video recordings\probe_YYYYMMDD_HHMMSS.mp4 --output-dir annotation_project --json-name annotations_train.json --class-name probe --keypoint-name tip --bbox-size-px 20 --step 5
```

Controls:
- slider: move through the video timeline
- left click: set tip keypoint on current frame
- right click: remove keypoint from current frame
- `a` / `d`: previous / next frame in slider timeline
- `s`: save COCO JSON + annotated frame images
- `q`: quit

Output:
- images: `annotation_project\images\frame_*.jpg`
- COCO JSON: `annotation_project\exports\annotations_train.json`

### Option B: External tool handoff

```powershell
python 04_annotation_handoff.py --frames-dir frames\<video_stem> --project-dir annotation_project --class-name probe --keypoint-name tip
```

Read `annotation_project\README_ANNOTATION.txt`, annotate externally, then export:
- `annotation_project\exports\annotations_train.json`
- `annotation_project\exports\annotations_val.json`

## 6. Convert COCO keypoints to YOLO pose labels

```powershell
python 05_convert_coco_to_yolo_pose.py --coco-json annotation_project\exports\annotations_train.json --labels-out labels_all --images-root frames\<video_stem> --class-id 0 --keypoint-index 0 --bbox-size-px 20
```

If you used `annotate.py`, set `--images-root annotation_project\images` instead.
If you have separate train and val exports, run converter for each into separate label folders and adjust split step accordingly.

## 7. Build dataset train/val split (80/20)

```powershell
python 06_split_dataset.py --images-dir annotation_project\images --labels-dir labels_all --dataset-dir dataset --val-ratio 0.2 --seed 42 --copy
```

If you used external annotation, set `--images-dir frames\<video_stem>` instead.

Resulting layout:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

## 8. Write Ultralytics data YAML

```powershell
python 07_write_data_yaml.py --dataset-dir dataset --yaml-path probe_tip.yaml --class-name probe
```

## 9. Train model

```powershell
python 08_train_pose.py --data-yaml probe_tip.yaml --model yolov8n-pose.pt --imgsz 640 --epochs 100 --batch 16 --name probe_tip_pose
```

Expected best weights:
- `runs\pose\probe_tip_pose\weights\best.pt`

## 10. Live classification (no filter)

```powershell
python 09_live_classify.py --weights runs\pose\probe_tip_pose\weights\best.pt --camera 0 --conf-thres 0.25
```

## 11. Live classification (Kalman filter)

```powershell
python 10_live_classify_kalman.py --weights runs\pose\probe_tip_pose\weights\best.pt --camera 0 --conf-thres 0.25 --process-var 2.0 --meas-var 36.0
```

## Notes

- You can annotate with the built-in `annotate.py` slider tool or external tools.
- Scripts use one class (`probe`) and one keypoint (`tip`).
- If COCO bbox is missing, converter synthesizes a `20x20` pixel box around the keypoint.

## Test Pipeline (Sequential Commands)

Run these in PowerShell, in repo root (`c:\Users\kleb\Documents\GitHub\Vision`).

1. Install dependencies and verify environment:

```powershell
python -m pip install -r requirements.txt
python 00_setup_env.py --check-camera --camera 0
```

2. Record a short demo video (press `r` to record, `q` to quit):

```powershell
python 01_record_video.py --camera 0 --output-dir recordings --width 1280 --height 720 --fps 30 --codec mp4v
```

3. Pick the newest recording into a shell variable:

```powershell
$VIDEO = (Get-ChildItem recordings\probe_*.mp4 | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$VIDEO
```

4. Optional review:

```powershell
python 02_review_video.py --video "$VIDEO"
```

5. Annotate with slider tool (press `s` to save, then `q`):

```powershell
python annotate.py --video "$VIDEO" --output-dir annotation_project --json-name annotations_train.json --class-name probe --keypoint-name tip --bbox-size-px 20 --step 5 --autosave-on-exit
```

6. Convert COCO keypoints JSON to YOLO labels:

```powershell
python 05_convert_coco_to_yolo_pose.py --coco-json annotation_project\exports\annotations_train.json --labels-out labels_all --images-root annotation_project\images --class-id 0 --keypoint-index 0 --bbox-size-px 20
```

7. Build train/val dataset split:

```powershell
python 06_split_dataset.py --images-dir annotation_project\images --labels-dir labels_all --dataset-dir dataset --val-ratio 0.2 --seed 42 --copy
```

8. Write data YAML:

```powershell
python 07_write_data_yaml.py --dataset-dir dataset --yaml-path probe_tip.yaml --class-name probe
```

9. Train a quick demo model:

```powershell
python 08_train_pose.py --data-yaml probe_tip.yaml --model yolov8n-pose.pt --imgsz 640 --epochs 10 --batch 8 --name probe_tip_pose_test --device cpu
```

10. Run live classification (no filter):

```powershell
python 09_live_classify.py --weights runs\pose\probe_tip_pose_test\weights\best.pt --camera 0 --conf-thres 0.25
```

11. Run live classification with Kalman filter:

```powershell
python 10_live_classify_kalman.py --weights runs\pose\probe_tip_pose_test\weights\best.pt --camera 0 --conf-thres 0.25 --process-var 2.0 --meas-var 36.0
```
