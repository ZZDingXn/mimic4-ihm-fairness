from __future__ import annotations

CHANNELS = [
    "Capillary refill rate",
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
]

CATEGORICAL_VALUES = {
    "Capillary refill rate": ["0.0", "1.0"],
    "Glascow coma scale eye opening": [
        "To Pain", "3 To speech", "1 No Response", "4 Spontaneously",
        "None", "To Speech", "Spontaneously", "2 To pain",
    ],
    "Glascow coma scale motor response": [
        "1 No Response", "3 Abnorm flexion", "Abnormal extension", "No response",
        "4 Flex-withdraws", "Localizes Pain", "Flex-withdraws", "Obeys Commands",
        "Abnormal Flexion", "6 Obeys Commands", "5 Localizes Pain", "2 Abnorm extensn",
    ],
    "Glascow coma scale total": [
        "11", "10", "13", "12", "15", "14", "3", "5", "4", "7", "6", "9", "8",
    ],
    "Glascow coma scale verbal response": [
        "1 No Response", "No Response", "Confused", "Inappropriate Words", "Oriented",
        "No Response-ETT", "5 Oriented", "Incomprehensible sounds", "1.0 ET/Trach",
        "4 Confused", "2 Incomp sounds", "3 Inapprop words",
    ],
}

NORMAL_VALUES = {
    "Capillary refill rate": "0.0",
    "Diastolic blood pressure": 59.0,
    "Fraction inspired oxygen": 0.21,
    "Glascow coma scale eye opening": "4 Spontaneously",
    "Glascow coma scale motor response": "6 Obeys Commands",
    "Glascow coma scale total": "15",
    "Glascow coma scale verbal response": "5 Oriented",
    "Glucose": 128.0,
    "Heart Rate": 86.0,
    "Height": 170.0,
    "Mean blood pressure": 77.0,
    "Oxygen saturation": 98.0,
    "Respiratory rate": 19.0,
    "Systolic blood pressure": 118.0,
    "Temperature": 36.6,
    "Weight": 81.0,
    "pH": 7.4,
}

CHART_ITEM_TO_CHANNEL = {
    224308: "Capillary refill rate",
    223951: "Capillary refill rate",
    220051: "Diastolic blood pressure",
    220180: "Diastolic blood pressure",
    224643: "Diastolic blood pressure",
    225310: "Diastolic blood pressure",
    227242: "Diastolic blood pressure",
    223835: "Fraction inspired oxygen",
    220739: "Glascow coma scale eye opening",
    223901: "Glascow coma scale motor response",
    223900: "Glascow coma scale verbal response",
    220621: "Glucose",
    225664: "Glucose",
    226537: "Glucose",
    220045: "Heart Rate",
    226707: "Height",
    226730: "Height",
    220052: "Mean blood pressure",
    220181: "Mean blood pressure",
    224322: "Mean blood pressure",
    225312: "Mean blood pressure",
    220227: "Oxygen saturation",
    220277: "Oxygen saturation",
    220210: "Respiratory rate",
    224422: "Respiratory rate",
    224689: "Respiratory rate",
    224690: "Respiratory rate",
    220050: "Systolic blood pressure",
    220179: "Systolic blood pressure",
    224167: "Systolic blood pressure",
    225309: "Systolic blood pressure",
    227243: "Systolic blood pressure",
    223761: "Temperature",
    223762: "Temperature",
    224639: "Weight",
    226512: "Weight",
    226531: "Weight",
    220274: "pH",
    223830: "pH",
}

LAB_ITEM_TO_CHANNEL = {
    50809: "Glucose",
    50931: "Glucose",
    50817: "Oxygen saturation",
    50820: "pH",
    50831: "pH",
}

EYE_CANONICAL = {
    "Spontaneously": ("4 Spontaneously", 4),
    "To Speech": ("3 To speech", 3),
    "To Pain": ("2 To pain", 2),
    "No Response": ("1 No Response", 1),
}
MOTOR_CANONICAL = {
    "Obeys Commands": ("6 Obeys Commands", 6),
    "Localizes Pain": ("5 Localizes Pain", 5),
    "Flex-withdraws": ("4 Flex-withdraws", 4),
    "Abnormal Flexion": ("3 Abnorm flexion", 3),
    "Abnormal extension": ("2 Abnorm extensn", 2),
    "No response": ("1 No Response", 1),
}
VERBAL_CANONICAL = {
    "Oriented": ("5 Oriented", 5),
    "Confused": ("4 Confused", 4),
    "Inappropriate Words": ("3 Inapprop words", 3),
    "Incomprehensible sounds": ("2 Incomp sounds", 2),
    "No Response": ("1 No Response", 1),
    "No Response-ETT": ("1.0 ET/Trach", 1),
}

GCS_SCORE = {}
for _channel, _mapping in (
    ("Glascow coma scale eye opening", EYE_CANONICAL),
    ("Glascow coma scale motor response", MOTOR_CANONICAL),
    ("Glascow coma scale verbal response", VERBAL_CANONICAL),
):
    for _canonical, _score in _mapping.values():
        GCS_SCORE[(_channel, _canonical)] = float(_score)


def build_header() -> tuple[list[str], dict[str, list[int]], list[int]]:
    header: list[str] = []
    channel_indices: dict[str, list[int]] = {}
    continuous_indices: list[int] = []
    for channel in CHANNELS:
        start = len(header)
        if channel in CATEGORICAL_VALUES:
            header.extend(f"{channel}->{value}" for value in CATEGORICAL_VALUES[channel])
        else:
            continuous_indices.append(len(header))
            header.append(channel)
        channel_indices[channel] = list(range(start, len(header)))
    for channel in CHANNELS:
        index = len(header)
        header.append(f"mask->{channel}")
        channel_indices[channel].append(index)
    if len(header) != 76:
        raise AssertionError(f"Expected 76 columns, got {len(header)}")
    return header, channel_indices, continuous_indices

