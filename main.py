import argparse

from ultralytics import YOLO

from config import load_config
from video_processor import process_video


def main(config_path):
    """
    Load config, initialize YOLO and process video.
    """

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config(
        config_path
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_path = config[
        "model_path"
    ]

    print(
        f"Loading model: {model_path}"
    )

    model = YOLO(
        model_path
    )

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    process_video(
        model,
        config,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Pose-based sitting/standing classifier"
        )
    )

    parser.add_argument(
        "config",
        help="Path to JSON configuration file",
    )

    args = parser.parse_args()

    main(
        args.config
    )