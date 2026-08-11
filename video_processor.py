from pathlib import Path

import cv2

from posture_classifier import classify_posture


# ============================================================
# OUTPUT PATH
# ============================================================

def get_unique_output_path(
    output_dir,
    video_path,
):
    """
    Generate unique output filename:

        video.avi
        video_1.avi
        video_2.avi
        ...
    """

    output_path = (
        output_dir
        / f"{video_path.stem}.avi"
    )

    if not output_path.exists():
        return output_path

    counter = 1

    while True:

        output_path = (
            output_dir
            / f"{video_path.stem}_{counter}.avi"
        )

        if not output_path.exists():
            return output_path

        counter += 1


# ============================================================
# ANNOTATION
# ============================================================

def annotate_frame(
    result,
    posture_config,
    visualization_config,
):
    """
    Draw YOLO pose, bbox and posture label.
    """

    show_bbox = visualization_config.get(
        "show_bbox",
        True,
    )

    show_pose = visualization_config.get(
        "show_pose",
        True,
    )

    show_posture = visualization_config.get(
        "show_posture",
        True,
    )

    annotated_frame = result.plot(
        boxes=show_bbox,
        labels=False,
        conf=False,
        kpt_line=show_pose,
        kpt_radius=3 if show_pose else 0,
    )

    if not show_posture:
        return annotated_frame

    if (
        result.keypoints is None
        or result.keypoints.xy is None
        or result.keypoints.conf is None
        or result.boxes is None
    ):
        return annotated_frame

    keypoints_all = (
        result.keypoints.xy
        .cpu()
        .numpy()
    )

    confidences_all = (
        result.keypoints.conf
        .cpu()
        .numpy()
    )

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    for (
        keypoints,
        confidences,
        box,
    ) in zip(
        keypoints_all,
        confidences_all,
        boxes,
    ):

        posture = classify_posture(
            keypoints,
            confidences,
            box,
            posture_config,
        )

        x1 = int(box[0])
        y1 = int(box[1])

        cv2.putText(
            annotated_frame,
            posture,
            (
                x1,
                max(y1 - 10, 25),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            posture_config.get(
                "text_scale",
                0.6,
            ),
            (255, 255, 255),
            posture_config.get(
                "text_thickness",
                2,
            ),
            cv2.LINE_AA,
        )

    return annotated_frame


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    model,
    config,
):
    """
    Process one video according to config.
    """

    video_path = Path(
        config["video_path"]
    )

    output_dir = Path(
        config["output_dir"]
    )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video does not exist: {video_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    yolo_config = config[
        "yolo"
    ]

    posture_config = config[
        "posture"
    ]

    visualization_config = config.get(
        "visualization",
        {},
    )

    # --------------------------------------------------------
    # YOLO PARAMETERS
    # --------------------------------------------------------

    confidence = yolo_config[
        "conf"
    ]

    iou = yolo_config.get(
        "iou",
        0.5,
    )

    image_size = yolo_config.get(
        "imgsz",
        640,
    )

    print(
        f"Video: {video_path}"
    )

    print(
        f"Classifier: "
        f"{posture_config['classifier_type']}"
    )

    # ========================================================
    # OPEN INPUT VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    output_path = get_unique_output_path(
        output_dir,
        video_path,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output: "
            f"{output_path}"
        )

    # ========================================================
    # PROCESS FRAMES
    # ========================================================

    frame_number = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            result = model(
                frame,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                verbose=False,
            )[0]

            annotated_frame = annotate_frame(
                result,
                posture_config,
                visualization_config,
            )

            writer.write(
                annotated_frame
            )

            frame_number += 1

            if frame_number % 100 == 0:

                print(
                    f"{frame_number}/"
                    f"{total_frames} frames"
                )

    finally:

        cap.release()
        writer.release()

    print(
        f"Saved: {output_path}"
    )

    return output_path