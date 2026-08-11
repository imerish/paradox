import numpy as np


# ============================================================
# COCO KEYPOINTS
# ============================================================

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_ELBOW = 7
RIGHT_ELBOW = 8

LEFT_WRIST = 9
RIGHT_WRIST = 10

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


BODY_SIDES = (
    (
        LEFT_SHOULDER,
        LEFT_HIP,
        LEFT_KNEE,
        LEFT_ANKLE,
    ),
    (
        RIGHT_SHOULDER,
        RIGHT_HIP,
        RIGHT_KNEE,
        RIGHT_ANKLE,
    ),
)


ARM_SIDES = (
    (
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST,
        LEFT_HIP,
    ),
    (
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST,
        RIGHT_HIP,
    ),
)


# ============================================================
# GEOMETRY
# ============================================================

def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.
    b is the vertex.
    """

    ba = (
        np.asarray(a, dtype=np.float32)
        - np.asarray(b, dtype=np.float32)
    )

    bc = (
        np.asarray(c, dtype=np.float32)
        - np.asarray(b, dtype=np.float32)
    )

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator == 0:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def calculate_distance(a, b):
    """
    Euclidean distance between two points.
    """

    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float32)
            - np.asarray(b, dtype=np.float32)
        )
    )


def is_visible(
    confidences,
    index,
    threshold,
):
    """
    Check whether keypoint confidence is high enough.
    """

    return confidences[index] >= threshold


def get_midpoint(
    keypoints,
    confidences,
    index1,
    index2,
    threshold,
):
    """
    If both points are visible:
        return midpoint.

    If only one is visible:
        return that point.

    If neither is visible:
        return None.
    """

    visible1 = is_visible(
        confidences,
        index1,
        threshold,
    )

    visible2 = is_visible(
        confidences,
        index2,
        threshold,
    )

    if visible1 and visible2:
        return (
            keypoints[index1]
            + keypoints[index2]
        ) / 2.0

    if visible1:
        return keypoints[index1]

    if visible2:
        return keypoints[index2]

    return None


# ============================================================
# NORMAL CLASSIFIER
# ============================================================

def classify_normal_posture(
    keypoints,
    confidences,
    config,
):
    """
    Normal-camera classifier.

    STANDING if at least one reliable body side has:

        shoulder -> hip -> knee >= hip threshold

    AND

        hip -> knee -> ankle >= knee threshold

    Otherwise:
        SITTING
    """

    confidence_threshold = config[
        "keypoint_conf"
    ]

    hip_threshold = config[
        "hip_standing_threshold"
    ]

    knee_threshold = config[
        "knee_standing_threshold"
    ]

    for (
        shoulder,
        hip,
        knee,
        ankle,
    ) in BODY_SIDES:

        required = (
            shoulder,
            hip,
            knee,
            ankle,
        )

        if not all(
            is_visible(
                confidences,
                index,
                confidence_threshold,
            )
            for index in required
        ):
            continue

        hip_angle = calculate_angle(
            keypoints[shoulder],
            keypoints[hip],
            keypoints[knee],
        )

        knee_angle = calculate_angle(
            keypoints[hip],
            keypoints[knee],
            keypoints[ankle],
        )

        if (
            hip_angle is None
            or knee_angle is None
        ):
            continue

        if (
            hip_angle >= hip_threshold
            and knee_angle >= knee_threshold
        ):
            return "STANDING"

    return "SITTING"


# ============================================================
# TOP-DOWN HELPERS
# ============================================================

def get_reliable_legs(
    confidences,
    threshold,
):
    """
    A leg is reliable when both knee and ankle
    have confidence >= threshold.
    """

    left = (
        is_visible(
            confidences,
            LEFT_KNEE,
            threshold,
        )
        and
        is_visible(
            confidences,
            LEFT_ANKLE,
            threshold,
        )
    )

    right = (
        is_visible(
            confidences,
            RIGHT_KNEE,
            threshold,
        )
        and
        is_visible(
            confidences,
            RIGHT_ANKLE,
            threshold,
        )
    )

    return left, right


def score_torso(
    keypoints,
    confidences,
    bbox_diagonal,
    config,
):
    """
    Calculate sitting/standing scores based on torso geometry.

    Returns:
        sitting_score,
        standing_score,
        torso_shoulder_ratio
    """

    threshold = config[
        "keypoint_conf"
    ]

    shoulder_mid = get_midpoint(
        keypoints,
        confidences,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        threshold,
    )

    hip_mid = get_midpoint(
        keypoints,
        confidences,
        LEFT_HIP,
        RIGHT_HIP,
        threshold,
    )

    if (
        shoulder_mid is None
        or hip_mid is None
    ):
        return 0.0, 0.0, None

    sitting_score = 0.0
    standing_score = 0.0

    torso_length = calculate_distance(
        shoulder_mid,
        hip_mid,
    )

    torso_shoulder_ratio = None

    both_shoulders_visible = (
        is_visible(
            confidences,
            LEFT_SHOULDER,
            threshold,
        )
        and
        is_visible(
            confidences,
            RIGHT_SHOULDER,
            threshold,
        )
    )

    # --------------------------------------------------------
    # TORSO / SHOULDER WIDTH
    # --------------------------------------------------------

    if both_shoulders_visible:

        shoulder_width = calculate_distance(
            keypoints[LEFT_SHOULDER],
            keypoints[RIGHT_SHOULDER],
        )

        if shoulder_width > 1.0:

            torso_shoulder_ratio = (
                torso_length
                / shoulder_width
            )

            if (
                torso_shoulder_ratio
                <= config[
                    "top_down_torso_shoulder_sitting"
                ]
            ):
                sitting_score += 4.0

            elif (
                torso_shoulder_ratio
                >= config[
                    "top_down_torso_shoulder_standing"
                ]
            ):
                standing_score += 3.0

            else:
                sitting_score += 0.5
                standing_score += 0.5

    # --------------------------------------------------------
    # TORSO / BBOX DIAGONAL
    # --------------------------------------------------------

    torso_bbox_ratio = (
        torso_length
        / bbox_diagonal
    )

    if (
        torso_bbox_ratio
        <= config[
            "top_down_torso_bbox_sitting"
        ]
    ):
        sitting_score += 1.5

    elif (
        torso_bbox_ratio
        >= config[
            "top_down_torso_bbox_standing"
        ]
    ):
        standing_score += 1.0

    return (
        sitting_score,
        standing_score,
        torso_shoulder_ratio,
    )


def score_arms(
    keypoints,
    confidences,
    bbox_diagonal,
    config,
):
    """
    Calculate sitting evidence from arm geometry.
    """

    threshold = config[
        "keypoint_conf"
    ]

    sitting_score = 0.0
    near_hip_count = 0

    for (
        shoulder,
        elbow,
        wrist,
        hip,
    ) in ARM_SIDES:

        shoulder_visible = is_visible(
            confidences,
            shoulder,
            threshold,
        )

        elbow_visible = is_visible(
            confidences,
            elbow,
            threshold,
        )

        wrist_visible = is_visible(
            confidences,
            wrist,
            threshold,
        )

        hip_visible = is_visible(
            confidences,
            hip,
            threshold,
        )

        # ----------------------------------------------------
        # ARM DISTANCE TO HIP
        # ----------------------------------------------------

        if hip_visible:

            distances = []

            if elbow_visible:
                distances.append(
                    calculate_distance(
                        keypoints[elbow],
                        keypoints[hip],
                    )
                )

            if wrist_visible:
                distances.append(
                    calculate_distance(
                        keypoints[wrist],
                        keypoints[hip],
                    )
                )

            if distances:

                ratio = (
                    min(distances)
                    / bbox_diagonal
                )

                if (
                    ratio
                    <= config[
                        "top_down_arm_hip_sitting_threshold"
                    ]
                ):
                    near_hip_count += 1

        # ----------------------------------------------------
        # ARM ANGLE
        # ----------------------------------------------------

        if (
            shoulder_visible
            and elbow_visible
            and wrist_visible
        ):

            angle = calculate_angle(
                keypoints[shoulder],
                keypoints[elbow],
                keypoints[wrist],
            )

            if (
                angle is not None
                and angle
                <= config[
                    "top_down_arm_sitting_max_angle"
                ]
            ):
                sitting_score += 0.5

    if near_hip_count == 1:
        sitting_score += 1.0

    elif near_hip_count >= 2:
        sitting_score += 2.0

    return sitting_score


def score_hips(
    keypoints,
    confidences,
    config,
):
    """
    Bent hip gives sitting evidence.

    Straight hip does not automatically
    give standing evidence.
    """

    threshold = config[
        "keypoint_conf"
    ]

    angles = []

    for (
        shoulder,
        hip,
        knee,
        _,
    ) in BODY_SIDES:

        if not (
            is_visible(
                confidences,
                shoulder,
                threshold,
            )
            and
            is_visible(
                confidences,
                hip,
                threshold,
            )
            and
            is_visible(
                confidences,
                knee,
                threshold,
            )
        ):
            continue

        angle = calculate_angle(
            keypoints[shoulder],
            keypoints[hip],
            keypoints[knee],
        )

        if angle is not None:
            angles.append(angle)

    if (
        angles
        and min(angles)
        <= config[
            "top_down_hip_sitting_threshold"
        ]
    ):
        return 2.0

    return 0.0


def score_knees(
    keypoints,
    confidences,
    left_reliable,
    right_reliable,
    config,
):
    """
    Straight reliable knee gives weak
    standing evidence.
    """

    threshold = config[
        "keypoint_conf"
    ]

    angles = []

    if (
        left_reliable
        and is_visible(
            confidences,
            LEFT_HIP,
            threshold,
        )
    ):

        angle = calculate_angle(
            keypoints[LEFT_HIP],
            keypoints[LEFT_KNEE],
            keypoints[LEFT_ANKLE],
        )

        if angle is not None:
            angles.append(angle)

    if (
        right_reliable
        and is_visible(
            confidences,
            RIGHT_HIP,
            threshold,
        )
    ):

        angle = calculate_angle(
            keypoints[RIGHT_HIP],
            keypoints[RIGHT_KNEE],
            keypoints[RIGHT_ANKLE],
        )

        if angle is not None:
            angles.append(angle)

    if (
        angles
        and max(angles)
        >= config[
            "top_down_knee_standing_threshold"
        ]
    ):
        return 0.75

    return 0.0


# ============================================================
# TOP-DOWN CLASSIFIER
# ============================================================

def classify_top_down_posture(
    keypoints,
    confidences,
    box,
    config,
):
    """
    Top-down classifier.

    Hard rule:
        no reliable leg -> SITTING
    """

    reliable_threshold = config[
        "top_down_reliable_leg_conf"
    ]

    (
        left_reliable,
        right_reliable,
    ) = get_reliable_legs(
        confidences,
        reliable_threshold,
    )

    # --------------------------------------------------------
    # HARD RULE
    # --------------------------------------------------------

    if not (
        left_reliable
        or right_reliable
    ):
        return "SITTING"

    # --------------------------------------------------------
    # BBOX
    # --------------------------------------------------------

    x1, y1, x2, y2 = box

    width = float(x2 - x1)
    height = float(y2 - y1)

    if (
        width <= 0
        or height <= 0
    ):
        return "SITTING"

    bbox_diagonal = float(
        np.hypot(
            width,
            height,
        )
    )

    if bbox_diagonal <= 0:
        return "SITTING"

    # --------------------------------------------------------
    # TORSO
    # --------------------------------------------------------

    (
        sitting_score,
        standing_score,
        torso_ratio,
    ) = score_torso(
        keypoints,
        confidences,
        bbox_diagonal,
        config,
    )

    # --------------------------------------------------------
    # ARMS
    # --------------------------------------------------------

    sitting_score += score_arms(
        keypoints,
        confidences,
        bbox_diagonal,
        config,
    )

    # --------------------------------------------------------
    # HIPS
    # --------------------------------------------------------

    sitting_score += score_hips(
        keypoints,
        confidences,
        config,
    )

    # --------------------------------------------------------
    # KNEES
    # --------------------------------------------------------

    standing_score += score_knees(
        keypoints,
        confidences,
        left_reliable,
        right_reliable,
        config,
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if (
        sitting_score
        >= config[
            "top_down_min_sitting_score"
        ]
        and sitting_score > standing_score
    ):
        return "SITTING"

    if (
        standing_score
        >= config[
            "top_down_min_standing_score"
        ]
        and standing_score > sitting_score
    ):
        return "STANDING"

    # --------------------------------------------------------
    # AMBIGUOUS FALLBACK
    # --------------------------------------------------------

    if (
        torso_ratio is not None
        and torso_ratio
        < config[
            "top_down_ambiguous_torso_ratio"
        ]
    ):
        return "SITTING"

    return "STANDING"


# ============================================================
# CLASSIFIER ROUTER
# ============================================================

def classify_posture(
    keypoints,
    confidences,
    box,
    config,
):
    """
    Select classifier based on classifier_type.
    """

    classifier_type = config[
        "classifier_type"
    ]

    if classifier_type == "normal":

        return classify_normal_posture(
            keypoints,
            confidences,
            config,
        )

    if classifier_type == "top_down":

        return classify_top_down_posture(
            keypoints,
            confidences,
            box,
            config,
        )

    raise ValueError(
        f"Unsupported classifier_type: "
        f"{classifier_type}"
    )