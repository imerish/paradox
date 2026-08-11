import json
from pathlib import Path


SUPPORTED_CLASSIFIERS = {
    "normal",
    "top_down",
}


def load_config(config_path):
    """
    Load and validate JSON configuration.
    """

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_fields = (
        "video_path",
        "model_path",
        "output_dir",
        "yolo",
        "posture",
    )

    missing_fields = [
        field
        for field in required_fields
        if field not in config
    ]

    if missing_fields:
        raise ValueError(
            "Missing config fields: "
            + ", ".join(missing_fields)
        )

    classifier_type = config["posture"].get(
        "classifier_type"
    )

    if classifier_type not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"Unsupported classifier_type: {classifier_type}. "
            f"Available: {', '.join(SUPPORTED_CLASSIFIERS)}"
        )

    return config