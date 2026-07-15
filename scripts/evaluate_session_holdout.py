import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_baseline_model import load_session, make_windows
from train_mlp_model import evaluate, standardize_train_test, to_target_label, train_mlp


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# učitava sesije odvojeno da svaka ostane cjelina u holdout podjeli
def collect_sessions(data_root, labels, output_labels, window_size, hop, binary_presence, normalize=False):
    output_index = {}
    for idx, label in enumerate(output_labels):
        output_index[label] = idx

    sessions = []
    for source_label in labels:
        for jsonl_path in sorted((data_root / source_label).glob("*/records.jsonl")):
            frames = load_session(jsonl_path)
            windows = make_windows(frames, window_size, hop, normalize=normalize)
            if not windows:
                continue
            mapped_label = to_target_label(source_label, binary_presence)
            sessions.append(
                {
                    "source_label": source_label,
                    "target_label": mapped_label,
                    "path": str(jsonl_path),
                    "frames": len(frames),
                    "windows": np.stack(windows).astype(np.float32),
                    "y": np.full(len(windows), output_index[mapped_label], dtype=np.int32),
                }
            )
    if len(sessions) < 2:
        raise SystemExit("Holdout procjena traži najmanje dvije sesije.")
    return sessions


# sažeti pregled sesija za rezultat
def summarize_sessions(sessions):
    summary = []
    for item in sessions:
        summary.append(
            {
                "source_label": item["source_label"],
                "target_label": item["target_label"],
                "path": item["path"],
                "frames": item["frames"],
                "windows": int(len(item["y"])),
            }
        )
    return summary


# spaja rezultate svih holdout prolaza u zbirne mjere
def aggregate(results, output_labels):
    confusion = defaultdict(Counter)
    total_windows = 0
    correct_windows = 0
    for result in results:
        total_windows += result["test_windows"]
        correct_windows += int(round(result["accuracy"] * result["test_windows"]))
        for true_label, predicted_counts in result["confusion"].items():
            for predicted_label, count in predicted_counts.items():
                confusion[true_label][predicted_label] += int(count)

    per_class = {}
    for label in output_labels:
        tp = confusion[label][label]
        # false positive: druge klase predviđene kao trenutna
        fp = 0
        for other in output_labels:
            if other != label:
                fp += confusion[other][label]
        # false negative: trenutna klasa predviđena kao druga
        fn = 0
        for other in output_labels:
            if other != label:
                fn += confusion[label][other]
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
        "accuracy": round(correct_windows / total_windows, 4) if total_windows else 0.0,
        "test_windows": total_windows,
        "per_class": per_class,
        "confusion": confusion_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Session-level holdout evaluation for CSI MLP models.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out", default="outputs/session_holdout_presence.json")
    parser.add_argument("--labels", nargs="+", default=["empty_room", "person_present_or_moving", "zone_door_left", "zone_middle", "zone_bed_right"])
    parser.add_argument("--multiclass", action="store_true", help="Zadržava izvorne oznake umjesto binarnog prisustva")
    parser.add_argument("--normalize-window", action="store_true", help="Normalizuje obilježja globalnim prosjekom prozora")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-holdouts", type=int, default=0, help="0 means evaluate every session.")
    args = parser.parse_args()

    source_labels = tuple(args.labels)
    binary_presence = not args.multiclass
    if args.multiclass:
        output_labels = source_labels
    else:
        output_labels = ("empty_room", "person_present")
    sessions = collect_sessions(Path(args.data_root), source_labels, output_labels, args.window_size, args.hop, binary_presence, normalize=args.normalize_window)
    if args.max_holdouts > 0:
        holdouts = sessions[: args.max_holdouts]
    else:
        holdouts = sessions
    results = []

    for holdout in holdouts:
        # jedna cijela sesija ostaje za test, a ostale ulaze u trening
        train_sessions = []
        for session in sessions:
            if session["path"] != holdout["path"]:
                train_sessions.append(session)

        train_windows = []
        train_y = []
        for session in train_sessions:
            train_windows.append(session["windows"])
            train_y.append(session["y"])
        x_train = np.concatenate(train_windows, axis=0)
        y_train = np.concatenate(train_y, axis=0)
        x_test = holdout["windows"]
        y_test = holdout["y"]
        x_train_std, x_test_std, _, _ = standardize_train_test(x_train, x_test)
        params, history = train_mlp(
            x_train_std,
            y_train,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed,
            use_balanced_weights=True,
        )
        metrics = evaluate(x_test_std, y_test, params, output_labels)
        results.append(
            {
                "holdout_path": holdout["path"],
                "source_label": holdout["source_label"],
                "target_label": holdout["target_label"],
                "test_windows": int(len(y_test)),
                "accuracy": metrics["accuracy"],
                "per_class": metrics["per_class"],
                "confusion": metrics["confusion"],
                "training_tail": history[-3:],
            }
        )

    artifact = {
        "created_utc": utc_now(),
        "mode": "session_level_holdout",
        "binary_presence": bool(binary_presence),
        "feature_norm": "window_mean" if args.normalize_window else "none",
        "window_size": args.window_size,
        "hop": args.hop,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "labels": list(output_labels),
        "sessions": summarize_sessions(sessions),
        "holdout_count": len(results),
        "aggregate": aggregate(results, output_labels),
        "results": results,
        "notes": [
            "Svaki prolaz izdvaja navedenu cijelu sesiju za test, a ostale koristi za trening.",
            "Podjela cijelih sesija daje realniju procjenu od nasumične podjele prozora.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "aggregate": artifact["aggregate"], "holdout_count": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
