# Sitting/Standing Posture Classification

## Introduction

This project detects people in video and classifies their posture as **SITTING** or **STANDING**.

It uses **YOLO11 Pose** to detect people and estimate COCO body keypoints such as shoulders, hips,
knees, ankles, elbows, and wrists. Posture is then determined using rule-based geometric analysis
of these keypoints, including joint angles, body proportions, keypoint confidence, and visibility.

Different configuration files are used to adjust the classification rules for different camera
perspectives, including normal and top-down views.

**OpenCV** is used for video reading, annotation, and saving the processed output.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Download Videos

Run the video download script:

```bash
python download_videos.py
```

## Run Posture Classification

Process the first video:

```bash
python main.py configs/video1.json
```

Process the second video:

```bash
python main.py configs/video2.json
```

The processed videos are saved in the `results/` directory.

## Configuration

Configuration files for each video are stored in:

```text
configs/
```

Each configuration file allows you to set:

- `video_path` — path to the input video
- `model_path` — path to the YOLO11 Pose model
- `output_dir` — directory for processed videos
- `yolo` — YOLO inference parameters such as `conf`, `iou`, and `imgsz`
- `posture` — posture classification parameters and classifier type (`normal` or `top_down`)
- `visualization` — output visualization options such as `show_bbox`, `show_pose`, and `show_posture`

## Project Structure

### `main.py`

Entry point of the application. Loads the configuration and YOLO model, then starts video processing.

### `posture_classifier.py`

Contains the rule-based posture classification logic for both normal and top-down camera perspectives.

### `video_processor.py`

Processes the input video frame by frame, runs YOLO pose estimation, classifies each detected person
as sitting or standing, draws the results, and saves the processed video.

### `config.py`

Loads and validates the JSON configuration file.

### `download_videos.py`

Downloads the input videos used for testing the posture classification pipeline.