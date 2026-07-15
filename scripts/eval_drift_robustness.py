import argparse
import json
from pathlib import Path

import numpy as np

from train_baseline_model import load_session, make_windows

# provjera otpornosti na drift amplitude
# globalni nivo CSI amplitude može porasti tokom rada i pomjeriti predikcije ka person_present
# drift se reprodukuje množenjem prazne sobe faktorom k i mjerenjem udjela empty predikcija
# normalizacija prosjekom prozora treba zadržati ravnu krivu kroz sve faktore


# udio prozora svrstanih u empty_room
def forward_empty_fraction(windows, model):
    labels = model["labels"]
    empty_idx = labels.index("empty_room")
    mean = np.asarray(model["scaler"]["mean"], dtype=np.float32)
    std = np.asarray(model["scaler"]["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    w1 = np.asarray(model["weights"]["w1"], dtype=np.float32)
    b1 = np.asarray(model["weights"]["b1"], dtype=np.float32)
    w2 = np.asarray(model["weights"]["w2"], dtype=np.float32)
    b2 = np.asarray(model["weights"]["b2"], dtype=np.float32)

    x = (windows - mean) / std
    h = np.maximum(x @ w1 + b1, 0.0)
    logits = h @ w2 + b2
    preds = logits.argmax(axis=1)
    return float((preds == empty_idx).mean())


def main():
    ap = argparse.ArgumentParser(description="Synthetic amplitude-drift robustness test for a presence MLP.")
    ap.add_argument("--model", action="append", required=True, help="Model JSON (repeatable).")
    ap.add_argument("--empty-records", required=True, help="records.jsonl of a captured empty room.")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--out", default="outputs/drift_robustness.json")
    args = ap.parse_args()

    frames = load_session(Path(args.empty_records))
    if len(frames) < args.window_size:
        raise SystemExit("Sesija prazne sobe nema dovoljno frejmova.")

    models = {}
    for path in args.model:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        models[path] = m

    report = {"empty_records": args.empty_records, "scales": args.scales, "models": {}}
    print(f"Empty-room drift test on {args.empty_records} ({len(frames)} frames)")

    # zaglavlje tabele ima jednu kolonu po modelu
    header_cols = []
    for p in args.model:
        header_cols.append(f"{Path(p).stem:>22}")
    print(f"{'scale k':>8} | " + " | ".join(header_cols))
    print("-" * (10 + 26 * len(args.model)))

    rows = {}
    for p in args.model:
        rows[p] = []
    for k in args.scales:
        # svaki frejm se množi faktorom k radi simulacije drifta
        scaled = []
        for f in frames:
            scaled.append(f * k)
        cells = []
        for p in args.model:
            m = models[p]
            normalize = m.get("feature_norm") == "window_mean"
            wins = make_windows(scaled, args.window_size, args.hop, normalize=normalize)
            frac = forward_empty_fraction(np.stack(wins).astype(np.float32), m)
            rows[p].append(frac)
            cells.append(f"{frac:>22.3f}")
        print(f"{k:>8.2f} | " + " | ".join(cells))

    for p in args.model:
        by_scale = {}
        for s, frac in zip(args.scales, rows[p]):
            by_scale[str(s)] = frac
        report["models"][p] = {
            "feature_norm": models[p].get("feature_norm", "none"),
            "empty_fraction_by_scale": by_scale,
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nValue = fraction of EMPTY windows still called empty_room (1.0 = perfect, ~0.0 = saturated to present).")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
