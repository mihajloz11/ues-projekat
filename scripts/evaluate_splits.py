import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_baseline_model import load_session, make_windows
from train_mlp_model import evaluate, split_indices, standardize_train_test, to_target_label, train_mlp


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# učitava svaku sesiju posebno za leave-one-session-out
def collect_sessions(data_root, source_labels, output_labels, window_size, hop, binary_presence):
    index = {}
    for i, label in enumerate(output_labels):
        index[label] = i
    sessions = []
    for source_label in source_labels:
        for jsonl in sorted((data_root / source_label).glob("*/records.jsonl")):
            frames = load_session(jsonl)
            wins = make_windows(frames, window_size, hop)
            if not wins:
                continue
            target = to_target_label(source_label, binary_presence)
            sessions.append({
                "source_label": source_label,
                "target_label": target,
                "name": jsonl.parent.name,
                "path": str(jsonl),
                "x": np.stack(wins).astype(np.float32),
                "y": np.full(len(wins), index[target], dtype=np.int32),
            })
    if len(sessions) < 2:
        raise SystemExit("Procjena traži najmanje dvije sesije.")
    return sessions


# zajednički parametri treninga
def train_kwargs(args):
    return dict(hidden_size=args.hidden_size, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, l2=args.l2, seed=args.seed, use_balanced_weights=True)


# nasumična podjela prozora je optimistična jer miješa prozore iste sesije
def window_level(sessions, output_labels, args):
    x_parts = []
    y_parts = []
    for s in sessions:
        x_parts.append(s["x"])
        y_parts.append(s["y"])
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    train_idx, test_idx = split_indices(y, args.test_ratio, args.seed)
    xtr_s, xte_s, _, _ = standardize_train_test(x[train_idx], x[test_idx])
    params, _ = train_mlp(xtr_s, y[train_idx], **train_kwargs(args))
    return evaluate(xte_s, y[test_idx], params, output_labels)


# poštena podjela izdvaja jednu cijelu sesiju za testiranje
def session_level(sessions, output_labels, args):
    confusion = defaultdict(Counter)
    per_session = []
    tot = 0
    cor = 0
    for ho in sessions:
        tr = []
        for s in sessions:
            if s["path"] != ho["path"]:
                tr.append(s)
        x_parts = []
        y_parts = []
        for s in tr:
            x_parts.append(s["x"])
            y_parts.append(s["y"])
        xtr = np.concatenate(x_parts)
        ytr = np.concatenate(y_parts)
        xtr_s, xte_s, _, _ = standardize_train_test(xtr, ho["x"])
        params, _ = train_mlp(xtr_s, ytr, **train_kwargs(args))
        m = evaluate(xte_s, ho["y"], params, output_labels)
        n = len(ho["y"])
        tot += n
        cor += int(round(m["accuracy"] * n))
        for t, pc in m["confusion"].items():
            for p, c in pc.items():
                confusion[t][p] += int(c)
        per_session.append({"name": ho["name"], "source_label": ho["source_label"],
                            "test_windows": n, "accuracy": m["accuracy"]})
    per_class = {}
    for label in output_labels:
        tp = confusion[label][label]
        fp = 0
        fn = 0
        for o in output_labels:
            if o != label:
                fp += confusion[o][label]
                fn += confusion[label][o]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    confusion_out = {}
    for t, c in confusion.items():
        confusion_out[t] = dict(c)
    return {
        "accuracy": round(cor / tot, 4) if tot else 0.0,
        "test_windows": tot,
        "per_class": per_class,
        "confusion": confusion_out,
        "per_session": per_session,
    }


def main():
    ap = argparse.ArgumentParser(description="Window vs session split comparison for a CSI task.")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="outputs/split_comparison.json")
    ap.add_argument("--labels", nargs="+",
                    default=["empty_room", "person_present_or_moving", "zone_door_left", "zone_middle", "zone_bed_right"])
    ap.add_argument("--multiclass", action="store_true", help="Zadržava izvorne oznake umjesto binarnog prisustva")
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--l2", type=float, default=0.0005)
    ap.add_argument("--test-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    binary = not args.multiclass
    source_labels = tuple(args.labels)
    if args.multiclass:
        output_labels = source_labels
    else:
        output_labels = ("empty_room", "person_present")
    sessions = collect_sessions(Path(args.data_root), source_labels, output_labels,
                                args.window_size, args.hop, binary)

    win = window_level(sessions, output_labels, args)
    ses = session_level(sessions, output_labels, args)
    # leakage gap pokazuje koliko nasumična podjela uljepšava rezultat
    gap = round(win["accuracy"] - ses["accuracy"], 4)

    print(f"\nTask: {'multiclass' if args.multiclass else 'binary presence'}   labels={output_labels}")
    print(f"sessions={len(sessions)}  window_size={args.window_size}  epochs={args.epochs}\n")
    print(f"  window-level (leaky)   accuracy = {win['accuracy']:.4f}")
    print(f"  session-level (honest) accuracy = {ses['accuracy']:.4f}")
    print(f"  LEAKAGE GAP            = {gap:.4f}\n")
    print("  per-session (honest):")
    for r in sorted(ses["per_session"], key=lambda d: d["accuracy"]):
        print(f"    {r['accuracy']:.3f}  {r['source_label']:26s} {r['name']}")

    artifact = {
        "created_utc": utc_now(),
        "task": "multiclass" if args.multiclass else "binary_presence",
        "labels": list(output_labels),
        "window_size": args.window_size, "hop": args.hop,
        "hidden_size": args.hidden_size, "epochs": args.epochs,
        "window_level": win,
        "session_level": ses,
        "leakage_gap": gap,
        "note": "leakage_gap = window-level minus session-level accuracy; large gap means the random split overstates real performance.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
