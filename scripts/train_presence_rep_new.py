import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csi_features as F
from compare_models import EMPTY_LABELS, OCCUPIED_LABELS, build_sessions, featurize

# trenira binarni MLP na REP_NEW obilježjima otpornim na drift
# model se čuva u istom JSON formatu sa W1/B1/W2/B2 težinama i scalerom
# prolaz unaprijed odgovara runtime postupku: ReLU pa softmax
# C runtime još koristi REP_OLD, pa ovaj model nije namijenjen trenutnom firmware-u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="models/presence_rep_new_mlp.json")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--max-win", type=int, default=300)
    ap.add_argument("--loso-acc", type=float, default=None, help="LOSO macro acc from compare_models, for record")
    args = ap.parse_args()

    from sklearn.neural_network import MLPClassifier

    sessions = build_sessions(Path(args.data_root))
    common_s = F.common_subcarriers(sessions)
    feats = featurize(sessions, F.REP_NEW, common_s, max_win=args.max_win)

    # spaja obilježja i oznake svih sesija koje imaju prozore
    X_parts = []
    y_parts = []
    for i in range(len(sessions)):
        if len(feats[i]):
            X_parts.append(feats[i])
            y_parts.append(np.full(len(feats[i]), sessions[i]["y"]))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    Xs = (X - mu) / sd

    clf = MLPClassifier(hidden_layer_sizes=(args.hidden,), activation="relu",
                        max_iter=600, random_state=0)
    clf.fit(Xs, y)
    train_acc = float(clf.score(Xs, y))

    w1, w2 = clf.coefs_[0], clf.coefs_[1]
    b1, b2 = clf.intercepts_[0], clf.intercepts_[1]
    # binarni sklearn izlaz pretvara se u softmax oblik sa dvije klase
    if w2.shape[1] == 1:
        w2 = np.hstack([-w2, w2])
        b2 = np.array([-b2[0], b2[0]])

    model = {
        "model_type": "amplitude_window_mlp",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": ["empty_room", "person_present"],
        "source_labels": EMPTY_LABELS + OCCUPIED_LABELS,
        "binary_presence": True,
        "feature_norm": "rep_new_ratio",
        "representation": {
            "name": "rep_new",
            "window_sec": 1.5, "hop_sec": 0.75,
            "per_subcarrier": ["coeff_of_variation", "motion_energy"],
            "common_subcarriers": int(common_s),
            "global_feats": 8,
        },
        "feature_count": int(X.shape[1]),
        "hidden_size": int(args.hidden),
        "activation": "relu",
        "output_activation": "softmax",
        "train_windows": int(X.shape[0]),
        "train_accuracy_in_sample": round(train_acc, 4),
        "loso_macro_accuracy": args.loso_acc,
        "scaler": {"mean": mu.tolist(), "std": sd.tolist()},
        "weights": {"w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(), "b2": b2.tolist()},
        "notes": [
            "Model je treniran na REP_NEW obilježjima otpornim na drift.",
            "Trenutni C runtime koristi REP_OLD, pa ovaj model nije usklađen za rad na uređaju.",
            "LOSO je mjerodavna procjena; tačnost trening skupa ne mjeri generalizaciju.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"feature_count={X.shape[1]} hidden={args.hidden} train_windows={X.shape[0]} "
          f"in_sample_acc={train_acc:.3f} loso={args.loso_acc}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
