import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CSI_RE = re.compile(r"\[(?P<values>[-0-9,\s]+)\]")
DEFAULT_LABELS = ("empty_room", "person_present_or_moving")

# najmanji dozvoljeni prosjek prozora pri normalizaciji
WINDOW_NORM_EPS = 1e-6


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# iz CSI linije izdvaja brojeve ili None za neispravnu liniju
def parse_csi_values(raw):
    match = CSI_RE.search(raw)
    if not match:
        return None
    values = []
    for part in match.group("values").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            return None
    return values or None


# iz I/Q parova računa amplitudu
def iq_to_amplitude(values):
    if len(values) % 2:
        values = values[:-1]
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    return np.sqrt((arr[:, 0] * arr[:, 0]) + (arr[:, 1] * arr[:, 1]))


# čita jednu JSONL sesiju i vraća amplitude po frejmu
def load_session(jsonl_path):
    frames = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("line_type") != "csi":
                continue
            values = parse_csi_values(record.get("raw", ""))
            if values:
                frames.append(iq_to_amplitude(values))
    return frames


# formira obilježja prozora: mean, std po podnosiocu i globalnu statistiku
# normalizacija prosjekom prozora poništava spori drift apsolutne amplitude
# C runtime koristi isti postupak, pa Python i pločica ostaju usklađeni
def make_windows(frames, window_size, hop, normalize=False):
    if len(frames) < window_size:
        return []
    min_len = min(len(frame) for frame in frames)
    trimmed = [frame[:min_len] for frame in frames]
    windows = []
    for start in range(0, len(trimmed) - window_size + 1, hop):
        chunk = np.stack(trimmed[start : start + window_size])
        mean = chunk.mean(axis=0)
        std = chunk.std(axis=0)
        global_features = np.asarray(
            [
                chunk.mean(),
                chunk.std(),
                chunk.min(),
                chunk.max(),
                np.median(chunk),
                np.percentile(chunk, 25),
                np.percentile(chunk, 75),
            ],
            dtype=np.float32,
        )
        vector = np.concatenate([mean, std, global_features]).astype(np.float32)
        if normalize:
            scale = float(chunk.mean())
            if scale > WINDOW_NORM_EPS:
                vector = (vector / scale).astype(np.float32)
        windows.append(vector)
    return windows


# prolazi kroz sesije i sastavlja matricu X, oznake y i pregled sesija
def collect_dataset(data_root, labels, window_size, hop):
    features = []
    y = []
    sessions = []
    for label_index, label in enumerate(labels):
        for jsonl_path in sorted((data_root / label).glob("*/records.jsonl")):
            frames = load_session(jsonl_path)
            windows = make_windows(frames, window_size, hop)
            sessions.append(
                {
                    "label": label,
                    "path": str(jsonl_path),
                    "frames": len(frames),
                    "windows": len(windows),
                }
            )
            features.extend(windows)
            y.extend([label_index] * len(windows))
    if not features:
        raise SystemExit("Nijedan trening prozor nije pronađen u data/raw/<label>/*/records.jsonl.")
    return np.stack(features), np.asarray(y, dtype=np.int32), sessions


# dijeli trening i test ravnomjerno po klasama
def split_indices(y, test_ratio, seed):
    rng = random.Random(seed)
    train_idx = []
    test_idx = []
    y_list = y.tolist()
    for label in sorted(set(y_list)):
        indices = []
        for idx, value in enumerate(y_list):
            if value == label:
                indices.append(idx)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_ratio)))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])
    return np.asarray(train_idx, dtype=np.int32), np.asarray(test_idx, dtype=np.int32)


# skalira na nultu sredinu i jediničnu devijaciju
def standardize_train_test(x_train, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


# centar klase je prosjek svih njenih primjera
def train_centroids(x, y, labels):
    centroids = {}
    for label_index, label in enumerate(labels):
        centroids[label] = x[y == label_index].mean(axis=0)
    return centroids


# bira klasu sa najbližim centrom
def predict_one(x, centroids, labels):
    distances = []
    for label in labels:
        distances.append(float(np.linalg.norm(x - centroids[label])))
    return int(np.argmin(distances))


# računa tačnost, matricu zabune i precision/recall/F1 po klasi
def evaluate(x, y, centroids, labels):
    preds_list = []
    for row in x:
        preds_list.append(predict_one(row, centroids, labels))
    preds = np.asarray(preds_list, dtype=np.int32)
    accuracy = float((preds == y).mean()) if len(y) else 0.0
    confusion = defaultdict(Counter)
    for true, pred in zip(y.tolist(), preds.tolist()):
        confusion[labels[true]][labels[pred]] += 1

    per_class = {}
    for idx, label in enumerate(labels):
        tp = int(((preds == idx) & (y == idx)).sum())
        fp = int(((preds == idx) & (y != idx)).sum())
        fn = int(((preds != idx) & (y == idx)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    confusion_out = {}
    for true, counts in confusion.items():
        confusion_out[true] = dict(counts)

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion": confusion_out,
    }


def to_float_list(arr, decimals=6):
    out = []
    for value in arr.tolist():
        out.append(round(float(value), decimals))
    return out


def main():
    parser = argparse.ArgumentParser(description="Train a simple WiFi CSI occupancy baseline.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out", default="models/baseline_csi_centroid.json")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--hop", type=int, default=8)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    args = parser.parse_args()

    labels = tuple(args.labels)
    x, y, sessions = collect_dataset(Path(args.data_root), labels, args.window_size, args.hop)
    train_idx, test_idx = split_indices(y, args.test_ratio, args.seed)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    x_train_std, x_test_std, scaler_mean, scaler_std = standardize_train_test(x_train, x_test)
    centroids = train_centroids(x_train_std, y_train, labels)
    train_metrics = evaluate(x_train_std, y_train, centroids, labels)
    test_metrics = evaluate(x_test_std, y_test, centroids, labels)

    # centroidi se pretvaraju u obične liste radi JSON zapisa
    centroids_out = {}
    for label, values in centroids.items():
        centroids_out[label] = to_float_list(values)

    artifact = {
        "model_type": "amplitude_window_nearest_centroid",
        "created_utc": utc_now(),
        "labels": list(labels),
        "window_size": args.window_size,
        "hop": args.hop,
        "feature_count": int(x.shape[1]),
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "sessions": sessions,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "scaler": {
            "mean": to_float_list(scaler_mean),
            "std": to_float_list(scaler_std),
        },
        "centroids": centroids_out,
        "notes": [
            "Početni model za provjeru CSI obilježja, a ne konačni model rada.",
            "Podjela je urađena po prozorima unutar sesija, pa rezultat može biti optimističan.",
            "Konačna procjena koristi odvojene sesije iz različitih dana i položaja.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "feature_count": artifact["feature_count"],
                "train_windows": artifact["train_windows"],
                "test_windows": artifact["test_windows"],
                "train_accuracy": train_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "test_confusion": test_metrics["confusion"],
                "sessions": sessions,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
