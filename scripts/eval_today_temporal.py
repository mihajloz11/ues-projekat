import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline_model import load_session, make_windows
from train_mlp_model import (
    evaluate,
    predict,
    standardize_train_test,
    to_target_label,
    train_mlp,
)

# hronološki holdout za podatke snimljene tokom jednog dana
# kada klasa ima samo jednu sesiju, LOSO bi uklonio cijelu negativnu ili pozitivnu klasu
# zato prvi dio svake sesije ulazi u trening, a vremenski kasniji dio u test


def main():
    ap = argparse.ArgumentParser(description="Chronological (temporal) holdout for a single-day CSI dataset.")
    ap.add_argument("--data-root", default="data/_today")
    ap.add_argument("--labels", nargs="+", default=["empty_room", "person_still", "walking"])
    ap.add_argument("--multiclass", action="store_true",
                    help="Zadržava izvorne oznake klasa umjesto binarnog prisustva")
    ap.add_argument("--normalize-window", action="store_true",
                    help="Dijeli obilježja prozora globalnim prosjekom amplitude radi uklanjanja drifta")
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--l2", type=float, default=0.0005)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/eval_today_temporal.json")
    args = ap.parse_args()

    if args.multiclass:
        output_labels = tuple(args.labels)
    else:
        output_labels = ("empty_room", "person_present")

    # sirovu oznaku preslikava u izlaznu klasu; u binarnom modu sve neprazno postaje present
    def map_label(s):
        if args.multiclass:
            return s
        return to_target_label(s, True)

    out_index = {}
    for i, lbl in enumerate(output_labels):
        out_index[lbl] = i

    Xtr, ytr, Xte, yte, src_te = [], [], [], [], []
    sessions = []

    for src in args.labels:
        for jl in sorted((Path(args.data_root) / src).glob("*/records.jsonl")):
            frames = load_session(jl)
            wins = make_windows(frames, args.window_size, args.hop, normalize=args.normalize_window)
            if not wins:
                continue
            wins = np.stack(wins).astype(np.float32)
            n = len(wins)
            k = int(n * args.train_frac)  # prvih k prozora je trening, a ostatak test
            yi = out_index[map_label(src)]
            Xtr.append(wins[:k])
            ytr.append(np.full(k, yi, dtype=np.int32))
            Xte.append(wins[k:])
            yte.append(np.full(n - k, yi, dtype=np.int32))
            src_te.extend([src] * (n - k))
            sessions.append({"label": src, "session": jl.parent.name,
                             "windows": n, "train": k, "test": n - k})

    if not Xtr:
        raise SystemExit(f"No windows under {args.data_root}. Check --labels / --data-root.")

    Xtr = np.concatenate(Xtr)
    ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte)
    yte = np.concatenate(yte)
    Xtr_s, Xte_s, _, _ = standardize_train_test(Xtr, Xte)

    params, _ = train_mlp(
        Xtr_s, ytr,
        hidden_size=args.hidden_size, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, l2=args.l2, seed=args.seed, use_balanced_weights=True,
    )
    metrics = evaluate(Xte_s, yte, params, output_labels)

    # tačnost se računa i po svakoj sirovoj oznaci
    preds = predict(Xte_s, params)
    by_src = defaultdict(lambda: {"n": 0, "correct": 0})
    for s, t, p in zip(src_te, yte.tolist(), preds.tolist()):
        by_src[s]["n"] += 1
        by_src[s]["correct"] += int(t == p)
    src_acc = {}
    for s, v in by_src.items():
        src_acc[s] = {"test_windows": v["n"], "detected": round(v["correct"] / v["n"], 4)}

    report = {
        "method": "chronological_holdout",
        "multiclass": bool(args.multiclass),
        "train_frac": args.train_frac,
        "window_size": args.window_size,
        "hop": args.hop,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "output_labels": list(output_labels),
        "train_windows": int(len(ytr)),
        "test_windows": int(len(yte)),
        "binary_accuracy": metrics["accuracy"],
        "per_class": metrics["per_class"],
        "confusion": metrics["confusion"],
        "per_source_label": src_acc,
        "sessions": sessions,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== Chronological holdout (train_frac={args.train_frac}) ===")
    print(f"train windows : {len(ytr):,}   test windows : {len(yte):,}")
    print(f"BINARY ACCURACY (empty vs present): {metrics['accuracy']*100:.2f}%")
    for lbl, m in metrics["per_class"].items():
        print(f"  {lbl:16s} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    print("  per-source detection (test tail):")
    for s, v in src_acc.items():
        print(f"    {s:24s} {v['detected']*100:6.2f}%  ({v['test_windows']:,} win)")
    print(f"  confusion: {metrics['confusion']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
