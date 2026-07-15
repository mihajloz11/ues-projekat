import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_baseline_model import load_session, make_windows


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# u binarnom modu sve što nije prazna soba postaje person_present
def to_target_label(source_label, binary_presence):
    if not binary_presence:
        return source_label
    if source_label == "empty_room":
        return "empty_room"
    return "person_present"


def relu(x):
    return np.maximum(x, 0.0)


def softmax(logits):
    # maksimum se oduzima radi numeričke stabilnosti
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


# spaja prozore svih sesija u matricu X i vektor oznaka y
def collect_dataset(data_root, source_labels, output_labels, window_size, hop, binary_presence, normalize=False):
    features = []
    y = []
    sessions = []

    # mapa oznake na indeks izlazne klase
    output_index = {}
    for idx, label in enumerate(output_labels):
        output_index[label] = idx

    for source_label in source_labels:
        for jsonl_path in sorted((data_root / source_label).glob("*/records.jsonl")):
            frames = load_session(jsonl_path)
            windows = make_windows(frames, window_size, hop, normalize=normalize)
            target_label = to_target_label(source_label, binary_presence)
            target_index = output_index[target_label]
            sessions.append(
                {
                    "source_label": source_label,
                    "target_label": target_label,
                    "path": str(jsonl_path),
                    "frames": len(frames),
                    "windows": len(windows),
                }
            )
            features.extend(windows)
            y.extend([target_index] * len(windows))

    if not features:
        raise SystemExit("Nijedan trening prozor nije pronađen u data/raw/<label>/*/records.jsonl.")
    return np.stack(features), np.asarray(y, dtype=np.int32), sessions


# dijeli trening i test ravnomjerno unutar svake klase
def split_indices(y, test_ratio, seed):
    rng = random.Random(seed)
    train_idx = []
    test_idx = []
    y_list = y.tolist()
    for label in sorted(set(y_list)):
        # indeksi svih primjera trenutne klase
        indices = []
        for idx, value in enumerate(y_list):
            if value == label:
                indices.append(idx)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_ratio)))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])
    return np.asarray(train_idx, dtype=np.int32), np.asarray(test_idx, dtype=np.int32)


# skaliranje na nultu sredinu i jediničnu devijaciju uči se iz trening skupa
def standardize_train_test(x_train, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def one_hot(y, class_count):
    out = np.zeros((len(y), class_count), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


# težine klasa sprečavaju zanemarivanje rjeđe klase
def class_weights(y, class_count):
    counts = np.bincount(y, minlength=class_count).astype(np.float32)
    total = float(counts.sum())
    weights = total / np.maximum(class_count * counts, 1.0)
    return weights.astype(np.float32)


# početne težine koriste He inicijalizaciju, a bias počinje od nule
def init_params(input_size, hidden_size, class_count, rng):
    w1 = rng.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, hidden_size)).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_size), size=(hidden_size, class_count)).astype(np.float32)
    b2 = np.zeros(class_count, dtype=np.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


# prolaz unaprijed koristi jedan skriveni sloj, ReLU i softmax
def forward(x, params):
    z1 = x @ params["w1"] + params["b1"]
    h1 = relu(z1)
    logits = h1 @ params["w2"] + params["b2"]
    probs = softmax(logits)
    return z1, h1, logits, probs


# gradijentni spust po mini-batchevima
def train_mlp(x, y, hidden_size, epochs, batch_size, lr, l2, seed, use_balanced_weights):
    rng = np.random.default_rng(seed)
    class_count = int(y.max()) + 1
    params = init_params(x.shape[1], hidden_size, class_count, rng)
    if use_balanced_weights:
        weights = class_weights(y, class_count)
    else:
        weights = np.ones(class_count, dtype=np.float32)
    y_one_hot = one_hot(y, class_count)
    indices = np.arange(len(y), dtype=np.int32)
    history = []

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            yb_oh = y_one_hot[batch_idx]
            sample_weights = weights[yb][:, None]

            z1, h1, _, probs = forward(xb, params)
            log_probs = np.log(np.maximum(probs, 1e-12))
            loss = -float((sample_weights * yb_oh * log_probs).sum() / len(batch_idx))
            loss += 0.5 * l2 * (float((params["w1"] ** 2).sum()) + float((params["w2"] ** 2).sum()))
            epoch_loss += loss * len(batch_idx)
            seen += len(batch_idx)

            # backprop prolazi kroz softmax i ReLU
            dlogits = (probs - yb_oh) * sample_weights / len(batch_idx)
            dw2 = h1.T @ dlogits + l2 * params["w2"]
            db2 = dlogits.sum(axis=0)
            dh1 = dlogits @ params["w2"].T
            dz1 = dh1 * (z1 > 0)
            dw1 = xb.T @ dz1 + l2 * params["w1"]
            db1 = dz1.sum(axis=0)

            params["w1"] -= lr * dw1.astype(np.float32)
            params["b1"] -= lr * db1.astype(np.float32)
            params["w2"] -= lr * dw2.astype(np.float32)
            params["b2"] -= lr * db2.astype(np.float32)

        # loss i tačnost se bilježe samo na odabranim epohama
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            preds = predict(x, params)
            accuracy = float((preds == y).mean())
            history.append({"epoch": epoch, "loss": round(epoch_loss / max(seen, 1), 6), "accuracy": round(accuracy, 4)})

    return params, history


def predict(x, params):
    _, _, _, probs = forward(x, params)
    return probs.argmax(axis=1).astype(np.int32)


# računa tačnost, matricu zabune i precision/recall/F1 po klasi
def evaluate(x, y, params, labels):
    preds = predict(x, params)
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

    # Counter matrica zabune pretvara se u običan rječnik radi JSON zapisa
    confusion_out = {}
    for true, counts in confusion.items():
        confusion_out[true] = dict(counts)

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion": confusion_out,
    }


def to_float_list(arr, decimals=6):
    return np.round(arr.astype(np.float32), decimals=decimals).tolist()


def main():
    parser = argparse.ArgumentParser(description="Train a small CSI MLP model.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out", default="models/zone_csi_mlp.json")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--hop", type=int, default=8)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", nargs="+", default=["empty_room", "zone_door_left", "zone_middle", "zone_bed_right"])
    parser.add_argument("--binary-presence", action="store_true", help="Map every non-empty label to person_present.")
    parser.add_argument(
        "--normalize-window",
        action="store_true",
        help="Divide each window feature vector by its global mean amplitude (cancels absolute-level drift).",
    )
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--no-balanced-weights", action="store_true")
    args = parser.parse_args()

    source_labels = tuple(args.labels)
    if args.binary_presence:
        output_labels = ("empty_room", "person_present")
    else:
        output_labels = source_labels
    x, y, sessions = collect_dataset(
        Path(args.data_root),
        source_labels,
        output_labels,
        args.window_size,
        args.hop,
        args.binary_presence,
        normalize=args.normalize_window,
    )
    train_idx, test_idx = split_indices(y, args.test_ratio, args.seed)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train_std, x_test_std, scaler_mean, scaler_std = standardize_train_test(x_train, x_test)

    params, history = train_mlp(
        x_train_std,
        y_train,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        use_balanced_weights=not args.no_balanced_weights,
    )

    train_metrics = evaluate(x_train_std, y_train, params, output_labels)
    test_metrics = evaluate(x_test_std, y_test, params, output_labels)
    artifact = {
        "model_type": "amplitude_window_mlp",
        "created_utc": utc_now(),
        "labels": list(output_labels),
        "source_labels": list(source_labels),
        "binary_presence": bool(args.binary_presence),
        "feature_norm": "window_mean" if args.normalize_window else "none",
        "window_size": args.window_size,
        "hop": args.hop,
        "feature_count": int(x.shape[1]),
        "hidden_size": args.hidden_size,
        "activation": "relu",
        "output_activation": "softmax",
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "sessions": sessions,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "l2": args.l2,
            "balanced_class_weights": not args.no_balanced_weights,
            "history": history,
        },
        "scaler": {
            "mean": to_float_list(scaler_mean),
            "std": to_float_list(scaler_std),
        },
        "weights": {
            "w1": to_float_list(params["w1"]),
            "b1": to_float_list(params["b1"]),
            "w2": to_float_list(params["w2"]),
            "b2": to_float_list(params["b2"]),
        },
        "notes": [
            "Mali NumPy MLP nad obilježjima prozora CSI amplitude.",
            "Metrike koriste stratifikovanu podjelu prozora i mogu biti optimistične.",
            "Konačni rezultat treba potvrditi holdout procjenom cijelih sesija.",
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
                "hidden_size": artifact["hidden_size"],
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
