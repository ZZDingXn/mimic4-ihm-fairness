from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy

from . import __version__
from .constants import (
    CATEGORICAL_VALUES,
    CHANNELS,
    CHART_ITEM_TO_CHANNEL,
    EYE_CANONICAL,
    LAB_ITEM_TO_CHANNEL,
    MOTOR_CANONICAL,
    NORMAL_VALUES,
    VERBAL_CANONICAL,
    build_header,
)
from .features import extract_lr_features, lr_feature_names


HOUR_NS = 3_600_000_000_000
WINDOW_HOURS = 48
DEMOGRAPHIC_MAPPING_VERSION = "mimic4-v3.1-r1"


def csv_path(root: Path, group: str, name: str) -> Path:
    candidates = [
        root / group / f"{name}.csv",
        root / group / f"{name}.csv.gz",
        root / f"{group}csv" / f"{name}.csv",
        root / f"{group}csv" / f"{name}.csv" / f"{name}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {group}/{name}.csv[.gz] under {root}")


def ethnicity_group(value: object) -> str:
    text = "" if pd.isna(value) else str(value).upper()
    if "HISPANIC" in text or "LATINO" in text or "SOUTH AMERICAN" in text:
        return "Hispanic"
    if "ASIAN" in text:
        return "Asian"
    if "BLACK" in text:
        return "Black"
    if "WHITE" in text:
        return "White"
    return "Other"


def insurance_group(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text == "medicare":
        return "Medicare"
    if text == "medicaid":
        return "Medicaid"
    if text == "private":
        return "Private"
    return "Other"


def age_group(age: float) -> str:
    if age < 30:
        return "18-29"
    if age < 50:
        return "30-49"
    if age < 70:
        return "50-69"
    if age < 90:
        return "70-89"
    return "90+"


def gender_group(value: object) -> str:
    return {"F": "Female", "M": "Male"}.get(str(value).upper(), "Other")


def _flow_row(stage: str, before: int, after: int) -> dict[str, int | str]:
    return {"stage": stage, "before": before, "excluded": before - after, "retained": after}


def select_cohort(root: Path) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    icu = pd.read_csv(csv_path(root, "icu", "icustays"), parse_dates=["intime", "outtime"])
    admissions = pd.read_csv(
        csv_path(root, "hosp", "admissions"),
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    patients = pd.read_csv(csv_path(root, "hosp", "patients"))
    flow: list[dict[str, int | str]] = []

    cohort = icu.merge(
        admissions[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "hospital_expire_flag",
                "insurance",
                "race",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="many_to_one",
    ).merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )
    flow.append(_flow_row("linked ICU stays", len(icu), len(cohort)))

    before = len(cohort)
    cohort["stays_per_admission"] = cohort.groupby("hadm_id")["stay_id"].transform("count")
    cohort = cohort[cohort["stays_per_admission"].eq(1)].copy()
    flow.append(_flow_row("one ICU stay per hospital admission", before, len(cohort)))

    before = len(cohort)
    cohort = cohort[cohort["first_careunit"].eq(cohort["last_careunit"])].copy()
    flow.append(_flow_row("no ICU unit transfer", before, len(cohort)))

    cohort["age"] = cohort["anchor_age"] + cohort["intime"].dt.year - cohort["anchor_year"]
    before = len(cohort)
    cohort = cohort[cohort["age"].ge(18)].copy()
    flow.append(_flow_row("adult age at least 18", before, len(cohort)))

    before = len(cohort)
    cohort = cohort[cohort["los"].ge(2.0)].copy()
    flow.append(_flow_row("ICU length of stay at least 48 hours", before, len(cohort)))

    cohort["death_within_exact_stay"] = (
        cohort["deathtime"].notna()
        & cohort["deathtime"].ge(cohort["admittime"])
        & cohort["deathtime"].le(cohort["dischtime"])
    ).astype(np.int8)
    cohort["death_within_admission_dates"] = (
        cohort["deathtime"].notna()
        & cohort["deathtime"].dt.normalize().ge(cohort["admittime"].dt.normalize())
        & cohort["deathtime"].dt.normalize().le(cohort["dischtime"].dt.normalize())
    ).astype(np.int8)
    # The formal protocol defines IHM from the recorded death timestamp lying
    # inside this hospital admission.  Keep hospital_expire_flag and the
    # date-only comparison as audit fields: MIMIC-IV can record death later on
    # the discharge date, so those definitions are expected to disagree for a
    # small, clinically meaningful set of admissions.
    cohort["label"] = cohort["death_within_exact_stay"].astype(np.int8)

    cohort["window_end"] = cohort["intime"] + pd.Timedelta(hours=WINDOW_HOURS)
    cohort["age_group"] = cohort["age"].map(age_group)
    cohort["gender_group"] = cohort["gender"].map(gender_group)
    cohort["ethnicity_group"] = cohort["race"].map(ethnicity_group)
    cohort["insurance_group"] = cohort["insurance"].map(insurance_group)
    return cohort.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True), flow


def read_subject_list(path: Path) -> set[int]:
    frame = pd.read_csv(path)
    if "subject_id" in frame:
        return set(frame["subject_id"].dropna().astype(int))
    if frame.shape[1] == 1:
        return set(frame.iloc[:, 0].dropna().astype(int))
    raise ValueError(f"{path} must contain a subject_id column or one unnamed column")


def deterministic_subset(cohort: pd.DataFrame, max_stays: int | None, seed: int) -> pd.DataFrame:
    if max_stays is None or len(cohort) <= max_stays:
        return cohort.copy()
    subjects = np.asarray(sorted(cohort["subject_id"].unique()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    chosen: list[int] = []
    count = 0
    sizes = cohort.groupby("subject_id")["stay_id"].size().to_dict()
    for subject_id in subjects:
        size = int(sizes[int(subject_id)])
        if chosen and count + size > max_stays:
            continue
        chosen.append(int(subject_id))
        count += size
        if count >= max_stays:
            break
    return cohort[cohort["subject_id"].isin(chosen)].copy()


def assign_splits(metadata: pd.DataFrame, seed: int) -> pd.Series:
    subjects = np.asarray(sorted(metadata["subject_id"].unique()), dtype=np.int64)
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for train/validation/test splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(round(0.15 * len(subjects))))
    remaining = len(subjects) - n_test
    n_validation = max(1, int(round(0.18 * remaining)))
    test = set(map(int, subjects[:n_test]))
    validation = set(map(int, subjects[n_test : n_test + n_validation]))
    mapping = {
        int(subject_id): ("test" if int(subject_id) in test else "validation" if int(subject_id) in validation else "train")
        for subject_id in subjects
    }
    return metadata["subject_id"].map(mapping)


def clean_chart_value(itemid: int, raw_value: object, valuenum: object) -> tuple[object | None, str | None]:
    channel = CHART_ITEM_TO_CHANNEL[itemid]
    text = "" if pd.isna(raw_value) else str(raw_value).strip()
    if channel == "Capillary refill rate":
        if text in {"Normal <3 Seconds", "Normal <3 secs", "Brisk"}:
            return "0.0", None
        if text in {"Abnormal >3 Seconds", "Abnormal >3 secs", "Delayed"}:
            return "1.0", None
        return None, f"unknown Capillary refill rate category: {text}"
    mappings = {
        "Glascow coma scale eye opening": EYE_CANONICAL,
        "Glascow coma scale motor response": MOTOR_CANONICAL,
        "Glascow coma scale verbal response": VERBAL_CANONICAL,
    }
    if channel in mappings:
        mapping = mappings[channel]
        mapped = mapping.get(text)
        if mapped:
            return mapped[0], None
        canonical_values = {canonical for canonical, _ in mapping.values()}
        if text in canonical_values:
            return text, None
        numeric_score = pd.to_numeric(valuenum, errors="coerce")
        if pd.isna(numeric_score):
            numeric_score = pd.to_numeric(raw_value, errors="coerce")
        if not pd.isna(numeric_score):
            for canonical, score in mapping.values():
                if float(numeric_score) == float(score):
                    return canonical, None
        return None, f"unknown {channel} category: {text}"

    value = pd.to_numeric(valuenum, errors="coerce")
    if pd.isna(value):
        value = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(value):
        return None, f"non-numeric {channel}: {text}"
    value = float(value)
    if channel == "Fraction inspired oxygen" and value > 1.0:
        value /= 100.0
    elif channel == "Oxygen saturation" and value <= 1.0:
        value *= 100.0
    elif itemid == 223761:
        value = (value - 32.0) * 5.0 / 9.0
    elif itemid == 226707:
        value = round(value * 2.54)
    elif itemid == 226531:
        value *= 0.453592
    return value, None


def initialise_database(path: Path, overwrite: bool) -> sqlite3.Connection:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; use --overwrite to replace generated artifacts")
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE events (
            stay_id INTEGER NOT NULL,
            charttime_ns INTEGER NOT NULL,
            channel TEXT NOT NULL,
            value_num REAL,
            value_text TEXT,
            itemid INTEGER NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    return connection


def _event_row(stay_id: int, charttime: pd.Timestamp, channel: str, value: object, itemid: int, source: str) -> tuple:
    if channel in CATEGORICAL_VALUES:
        return int(stay_id), int(charttime.value), channel, None, str(value), int(itemid), source
    return int(stay_id), int(charttime.value), channel, float(value), None, int(itemid), source


def stage_chart_events(
    root: Path,
    cohort: pd.DataFrame,
    connection: sqlite3.Connection,
    chunksize: int,
    audit: Counter,
    unknown: Counter,
    resources: list[dict[str, int | str]],
) -> None:
    lookup = cohort.set_index("stay_id")[["intime", "window_end"]]
    path = csv_path(root, "icu", "chartevents")
    usecols = ["stay_id", "charttime", "itemid", "value", "valuenum"]
    sql = "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)"
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, parse_dates=["charttime"], chunksize=chunksize), start=1
    ):
        read_rows = len(chunk)
        chunk = chunk[
            chunk["stay_id"].isin(lookup.index) & chunk["itemid"].isin(CHART_ITEM_TO_CHANNEL)
        ].copy()
        if chunk.empty:
            resources.append(resource_snapshot("chartevents", chunk_number, read_rows, 0, 0))
            print(f"chartevents chunk {chunk_number}: staged 0 rows", flush=True)
            continue
        chunk = chunk.join(lookup, on="stay_id")
        chunk = chunk[
            chunk["charttime"].ge(chunk["intime"])
            & chunk["charttime"].le(chunk["window_end"])
        ]
        rows = []
        for row in chunk.itertuples(index=False):
            itemid = int(row.itemid)
            audit[("chart", itemid, "seen")] += 1
            value, reason = clean_chart_value(itemid, row.value, row.valuenum)
            if value is None:
                unknown[reason or "unknown chart cleaning failure"] += 1
                continue
            rows.append(_event_row(row.stay_id, row.charttime, CHART_ITEM_TO_CHANNEL[itemid], value, itemid, "chart"))
            audit[("chart", itemid, "kept")] += 1
        if rows:
            connection.executemany(sql, rows)
            connection.commit()
        resources.append(resource_snapshot("chartevents", chunk_number, read_rows, len(chunk), len(rows)))
        print(f"chartevents chunk {chunk_number}: staged {len(rows):,} rows", flush=True)


def stage_lab_events(
    root: Path,
    cohort: pd.DataFrame,
    connection: sqlite3.Connection,
    chunksize: int,
    audit: Counter,
    unknown: Counter,
    resources: list[dict[str, int | str]],
) -> None:
    lookup = cohort.set_index("hadm_id")[["stay_id", "intime", "window_end"]]
    path = csv_path(root, "hosp", "labevents")
    usecols = ["hadm_id", "charttime", "itemid", "value", "valuenum"]
    sql = "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)"
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, parse_dates=["charttime"], chunksize=chunksize), start=1
    ):
        read_rows = len(chunk)
        chunk = chunk[
            chunk["hadm_id"].isin(lookup.index) & chunk["itemid"].isin(LAB_ITEM_TO_CHANNEL)
        ].copy()
        if chunk.empty:
            resources.append(resource_snapshot("labevents", chunk_number, read_rows, 0, 0))
            print(f"labevents chunk {chunk_number}: staged 0 rows", flush=True)
            continue
        chunk = chunk.join(lookup, on="hadm_id")
        chunk = chunk[
            chunk["charttime"].ge(chunk["intime"])
            & chunk["charttime"].le(chunk["window_end"])
        ]
        rows = []
        for row in chunk.itertuples(index=False):
            itemid = int(row.itemid)
            channel = LAB_ITEM_TO_CHANNEL[itemid]
            audit[("lab", itemid, "seen")] += 1
            value = pd.to_numeric(row.valuenum, errors="coerce")
            if pd.isna(value):
                value = pd.to_numeric(row.value, errors="coerce")
            if pd.isna(value):
                unknown[f"non-numeric lab {channel}: {row.value}"] += 1
                continue
            value = float(value)
            if channel == "Oxygen saturation" and value <= 1.0:
                value *= 100.0
            rows.append(_event_row(row.stay_id, row.charttime, channel, value, itemid, "lab"))
            audit[("lab", itemid, "kept")] += 1
        if rows:
            connection.executemany(sql, rows)
            connection.commit()
        resources.append(resource_snapshot("labevents", chunk_number, read_rows, len(chunk), len(rows)))
        print(f"labevents chunk {chunk_number}: staged {len(rows):,} rows", flush=True)


def resource_snapshot(source: str, chunk: int, rows_read: int, rows_selected: int, rows_staged: int) -> dict[str, int | str]:
    memory = psutil.Process().memory_info()
    return {
        "source": source,
        "chunk": chunk,
        "rows_read": rows_read,
        "rows_selected_in_window": rows_selected,
        "rows_staged": rows_staged,
        "rss_bytes": int(memory.rss),
        "vms_bytes": int(memory.vms),
    }


def add_gcs_total(events: list[tuple[int, str, object, int, str]]) -> list[tuple[int, str, object, int, str]]:
    component_channels = {
        "Glascow coma scale eye opening": EYE_CANONICAL,
        "Glascow coma scale motor response": MOTOR_CANONICAL,
        "Glascow coma scale verbal response": VERBAL_CANONICAL,
    }
    scores: dict[tuple[int, str], int] = {}
    for time_ns, channel, value, _, _ in events:
        if channel not in component_channels:
            continue
        mapping = component_channels[channel]
        raw_to_score = {canonical: score for canonical, score in mapping.values()}
        if str(value) in raw_to_score:
            scores[(time_ns, channel)] = raw_to_score[str(value)]
    times = sorted({time_ns for time_ns, _ in scores})
    derived = []
    for time_ns in times:
        values = [scores.get((time_ns, channel)) for channel in component_channels]
        if all(value is not None for value in values):
            derived.append((time_ns, "Glascow coma scale total", str(sum(values)), -1, "derived"))
    return sorted(events + derived, key=lambda row: (row[0], row[3]))


def encode_stay(
    events: list[tuple[int, str, object, int, str]], intime: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header, _, _ = build_header()
    feature_width = len(header) - len(CHANNELS)
    values = np.zeros((WINDOW_HOURS, feature_width), dtype=np.float32)
    masks = np.zeros((WINDOW_HOURS, len(CHANNELS)), dtype=np.float32)
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for channel in CHANNELS:
        width = len(CATEGORICAL_VALUES.get(channel, [None]))
        offsets[channel] = (cursor, cursor + width)
        cursor += width

    lr_events: list[tuple[float, str, object]] = []
    for time_ns, channel, value, _, _ in events:
        hour = (time_ns - intime.value) / HOUR_NS
        if hour < -1e-6 or hour > WINDOW_HOURS + 1e-6:
            continue
        hour = min(max(hour, 0.0), float(WINDOW_HOURS))
        bin_id = min(int(max(hour - 1e-6, 0.0)), WINDOW_HOURS - 1)
        channel_id = CHANNELS.index(channel)
        begin, end = offsets[channel]
        if channel in CATEGORICAL_VALUES:
            text = str(value)
            if text not in CATEGORICAL_VALUES[channel]:
                continue
            values[bin_id, begin:end] = 0.0
            values[bin_id, begin + CATEGORICAL_VALUES[channel].index(text)] = 1.0
        else:
            values[bin_id, begin] = float(value)
        masks[bin_id, channel_id] = 1.0
        lr_events.append((hour, channel, value))

    for channel_id, channel in enumerate(CHANNELS):
        begin, end = offsets[channel]
        if channel in CATEGORICAL_VALUES:
            normal = np.zeros(end - begin, dtype=np.float32)
            normal[CATEGORICAL_VALUES[channel].index(str(NORMAL_VALUES[channel]))] = 1.0
        else:
            normal = np.asarray([NORMAL_VALUES[channel]], dtype=np.float32)
        previous = normal
        for timestep in range(WINDOW_HOURS):
            if masks[timestep, channel_id]:
                previous = values[timestep, begin:end].copy()
            else:
                values[timestep, begin:end] = previous
    encoded = np.concatenate([values, masks], axis=1)
    return encoded, masks, extract_lr_features(lr_events)


def _database_events(connection: sqlite3.Connection):
    query = (
        "SELECT stay_id, charttime_ns, channel, value_num, value_text, itemid, source "
        "FROM events ORDER BY stay_id, charttime_ns, itemid"
    )
    for stay_id, rows in groupby(connection.execute(query), key=lambda row: int(row[0])):
        events = [
            (int(row[1]), str(row[2]), row[4] if row[4] is not None else float(row[3]), int(row[5]), str(row[6]))
            for row in rows
        ]
        yield stay_id, add_gcs_total(events)


def _normalise_lstm(path: Path, metadata: pd.DataFrame, continuous_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.load(path, mmap_mode="r+")
    train_rows = np.flatnonzero(metadata["split"].eq("train").to_numpy())
    sums = np.zeros(len(continuous_indices), dtype=np.float64)
    sums_sq = np.zeros(len(continuous_indices), dtype=np.float64)
    count = 0
    for index in train_rows:
        values = np.asarray(matrix[index], dtype=np.float64)[:, continuous_indices]
        sums += values.sum(axis=0)
        sums_sq += np.square(values).sum(axis=0)
        count += values.shape[0]
    means = sums / count
    variances = np.maximum(sums_sq / count - np.square(means), 1e-14)
    stds = np.sqrt(variances)
    for start in range(0, len(matrix), 256):
        stop = min(start + 256, len(matrix))
        block = np.asarray(matrix[start:stop]).copy()
        block[:, :, continuous_indices] = (block[:, :, continuous_indices] - means) / stds
        matrix[start:stop] = block
        matrix.flush()
    return means, stds


def write_mapping_audit(root: Path, output: Path, audit: Counter) -> None:
    chart_items = pd.read_csv(csv_path(root, "icu", "d_items"), dtype={"itemid": int})
    lab_items = pd.read_csv(csv_path(root, "hosp", "d_labitems"), dtype={"itemid": int})
    rows = []
    for source, mapping, dictionary in (
        ("chart", CHART_ITEM_TO_CHANNEL, chart_items),
        ("lab", LAB_ITEM_TO_CHANNEL, lab_items),
    ):
        labels = dictionary.set_index("itemid")
        for itemid, channel in sorted(mapping.items()):
            record = labels.loc[itemid] if itemid in labels.index else None
            rows.append(
                {
                    "source": source,
                    "itemid": itemid,
                    "channel": channel,
                    "label": "" if record is None else record.get("label", ""),
                    "unitname": "" if record is None else record.get("unitname", ""),
                    "seen_in_window": audit[(source, itemid, "seen")],
                    "kept_after_cleaning": audit[(source, itemid, "kept")],
                }
            )
    pd.DataFrame(rows).to_csv(output, index=False)


def write_demographic_mapping_audit(metadata: pd.DataFrame, output: Path) -> None:
    specifications = (
        ("gender", "gender_group", "exact F/M; all other or missing values -> Other"),
        (
            "race",
            "ethnicity_group",
            "priority substring rules: HISPANIC/LATINO/SOUTH AMERICAN, ASIAN, BLACK, WHITE; otherwise Other",
        ),
        ("insurance", "insurance_group", "case-insensitive exact Medicare/Medicaid/Private; otherwise Other"),
    )
    rows = []
    for raw_column, group_column, rule in specifications:
        frame = metadata[[raw_column, group_column]].copy()
        frame[raw_column] = frame[raw_column].fillna("<MISSING>").astype(str)
        counts = frame.groupby([raw_column, group_column], dropna=False).size()
        for (raw_value, mapped_group), count in counts.items():
            rows.append(
                {
                    "mapping_version": DEMOGRAPHIC_MAPPING_VERSION,
                    "attribute": raw_column,
                    "raw_value": raw_value,
                    "mapped_group": mapped_group,
                    "count": int(count),
                    "rule": rule,
                }
            )
    pd.DataFrame(rows).sort_values(["attribute", "mapped_group", "raw_value"]).to_csv(output, index=False)


def write_continuous_value_audit(connection: sqlite3.Connection, output: Path) -> None:
    query = """
        SELECT source, itemid, channel, COUNT(*) AS count,
               MIN(value_num) AS minimum, MAX(value_num) AS maximum,
               AVG(value_num) AS mean
        FROM events
        WHERE value_num IS NOT NULL
        GROUP BY source, itemid, channel
        ORDER BY channel, source, itemid
    """
    pd.read_sql_query(query, connection).to_csv(output, index=False)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort, flow = select_cohort(args.mimic_root)
    if args.subject_list:
        allowed = read_subject_list(args.subject_list)
        before = len(cohort)
        cohort = cohort[cohort["subject_id"].isin(allowed)].copy()
        flow.append(_flow_row("explicit subject list", before, len(cohort)))
    before = len(cohort)
    cohort = deterministic_subset(cohort, args.max_stays, args.seed)
    flow.append(_flow_row("deterministic study subset", before, len(cohort)))
    if cohort.empty:
        raise RuntimeError("Cohort is empty before event extraction")

    connection = initialise_database(output / "events.sqlite", args.overwrite)
    audit: Counter = Counter()
    unknown: Counter = Counter()
    resources: list[dict[str, int | str]] = []
    stage_chart_events(args.mimic_root, cohort, connection, args.chunksize, audit, unknown, resources)
    stage_lab_events(args.mimic_root, cohort, connection, args.chunksize, audit, unknown, resources)
    pd.DataFrame(resources).to_csv(output / "resource_usage.csv", index=False)
    connection.execute("CREATE INDEX events_stay_time ON events(stay_id, charttime_ns,itemid)")
    connection.commit()

    event_stays = {int(row[0]) for row in connection.execute("SELECT DISTINCT stay_id FROM events")}
    before = len(cohort)
    cohort = cohort[cohort["stay_id"].isin(event_stays)].copy()
    flow.append(_flow_row("at least one usable observation in first 48 hours", before, len(cohort)))
    cohort = cohort.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)
    cohort["split"] = assign_splits(cohort, args.seed)
    stay_to_row = {int(stay_id): index for index, stay_id in enumerate(cohort["stay_id"])}

    header, channel_indices, continuous_indices = build_header()
    x_path = output / "lstm_X.npy"
    lr_path = output / "lr_X_raw.npy"
    if (x_path.exists() or lr_path.exists()) and not args.overwrite:
        raise FileExistsError("Prepared arrays already exist; use --overwrite to replace generated artifacts")
    x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(len(cohort), 48, 76))
    lr = np.lib.format.open_memmap(lr_path, mode="w+", dtype=np.float32, shape=(len(cohort), 714))
    observed_counts = np.zeros((len(cohort), len(CHANNELS)), dtype=np.float32)

    intimes = cohort.set_index("stay_id")["intime"].to_dict()
    encoded_count = 0
    for stay_id, events in _database_events(connection):
        if stay_id not in stay_to_row:
            continue
        index = stay_to_row[stay_id]
        encoded, masks, lr_features = encode_stay(events, intimes[stay_id])
        x[index] = encoded
        lr[index] = lr_features
        observed_counts[index] = masks.sum(axis=0)
        encoded_count += 1
    x.flush()
    lr.flush()
    if encoded_count != len(cohort):
        raise AssertionError(f"Encoded {encoded_count} stays but cohort contains {len(cohort)}")

    means, stds = _normalise_lstm(x_path, cohort, continuous_indices)
    np.save(output / "y.npy", cohort["label"].to_numpy(np.int8))
    np.save(output / "split.npy", cohort["split"].to_numpy(str))
    cohort.to_csv(output / "metadata.csv", index=False)
    write_demographic_mapping_audit(cohort, output / "demographic_mapping_audit.csv")
    label_audit_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
        "death_within_exact_stay",
        "death_within_admission_dates",
    ]
    label_mismatch = cohort["hospital_expire_flag"].ne(cohort["death_within_exact_stay"])
    cohort.loc[label_mismatch, label_audit_columns].to_csv(output / "label_audit.csv", index=False)
    cohort[["subject_id", "split"]].drop_duplicates().sort_values("subject_id").to_csv(
        output / "split_manifest.csv", index=False
    )
    pd.DataFrame(flow).to_csv(output / "cohort_flow.csv", index=False)
    coverage = pd.DataFrame(
        {
            "channel": CHANNELS,
            "none_percent": 100 * np.mean(observed_counts == 0, axis=0),
            "full_percent": 100 * np.mean(observed_counts == 48, axis=0),
            "mean_observed_hours": observed_counts.mean(axis=0),
            "std_observed_hours": observed_counts.std(axis=0, ddof=1),
        }
    )
    coverage.to_csv(output / "coverage.csv", index=False)
    write_mapping_audit(args.mimic_root, output / "mapping_audit.csv", audit)
    write_continuous_value_audit(connection, output / "continuous_value_audit.csv")
    pd.DataFrame(
        [{"issue": issue, "count": count} for issue, count in unknown.most_common()],
        columns=["issue", "count"],
    ).to_csv(output / "unknown_values.csv", index=False)

    config = {
        "schema_version": 1,
        "package_version": __version__,
        "header": header,
        "channels": CHANNELS,
        "channel_indices": channel_indices,
        "continuous_indices": continuous_indices,
        "normalization_mean": means.tolist(),
        "normalization_std": stds.tolist(),
        "lr_feature_names": lr_feature_names(),
        "timestep_hours": 1.0,
        "window_hours": WINDOW_HOURS,
        "seed": args.seed,
        "max_stays": args.max_stays,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    provenance = {
        "command": sys.argv,
        "mimic_root": str(args.mimic_root.resolve()),
        "input_files": {
            f"{group}/{name}": {
                "path": str(csv_path(args.mimic_root, group, name).resolve()),
                "size": csv_path(args.mimic_root, group, name).stat().st_size,
            }
            for group, name in (
                ("hosp", "patients"),
                ("hosp", "admissions"),
                ("hosp", "d_labitems"),
                ("hosp", "labevents"),
                ("icu", "icustays"),
                ("icu", "d_items"),
                ("icu", "chartevents"),
            )
        },
        "git_commit": git_commit(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "psutil": psutil.__version__,
        "peak_observed_rss_bytes": max((int(row["rss_bytes"]) for row in resources), default=None),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    connection.close()
    print(
        cohort.groupby("split")["label"].agg(["size", "sum", "mean"]).rename(columns={"sum": "deaths"})
    )
    print(f"Prepared {len(cohort):,} stays at {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MIMIC-IV v3.1 IHM data for LSTM and LR")
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=49297)
    parser.add_argument("--max-stays", type=int)
    parser.add_argument("--subject-list", type=Path)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
