import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csi_features as F

warnings.filterwarnings("ignore")

# pošteno poređenje modela i reprezentacija za binarno prisustvo
# leave-one-session-out (LOSO) drži povezane prozore iste sesije zajedno

EMPTY_LABELS = ["empty_room"]
OCCUPIED_LABELS = [
    "walking", "person_present_or_moving", "person_still",
    "zone_bed_right", "zone_desk_chair", "zone_door_left", "zone_middle",
]

# odvojeni zadaci pokazuju razliku između signala pokreta i mirovanja
TASKS = {
    "presence_all": OCCUPIED_LABELS,
    "moving_vs_empty": ["walking", "person_present_or_moving"],
    "still_vs_empty": ["person_still", "zone_bed_right", "zone_desk_chair",
                       "zone_door_left", "zone_middle"],
}


# učitava sve sesije i označava ih kao prazno=0 ili zauzeto=1
def build_sessions(data_root, occ_labels=None):
    if not occ_labels:
        occ_labels = OCCUPIED_LABELS
    pairs = []
    for l in EMPTY_LABELS:
        pairs.append((l, 0))
    for l in occ_labels:
        pairs.append((l, 1))
    sessions = []
    for label, y in pairs:
        for jsonl in sorted((data_root / label).glob("*/records.jsonl")):
            s = F.load_session(jsonl)
            if s is None:
                continue
            fps = F.effective_fps(jsonl.parent, s["amp"].shape[0], s.get("local_ts"))
            sessions.append({
                "id": f"{label}/{jsonl.parent.name}",
                "label": label, "y": y, "amp": s["amp"], "fps": fps,
            })
    return sessions


# formira matricu obilježja za svaku sesiju i izabranu reprezentaciju
def featurize(sessions, rep, common_s, max_win=0):
    feats = []
    for s in sessions:
        amp = s["amp"][:, :common_s]
        if rep == F.REP_OLD:
            X = F.windows_old(amp)
        else:
            X = F.windows_new(amp, s["fps"])
        if max_win and len(X) > max_win:
            idx = np.linspace(0, len(X) - 1, max_win).astype(int)  # ravnomjerno prorjeđivanje
            X = X[idx]
        feats.append(X)
    return feats


# sklearn modeli koji ulaze u poređenje
def make_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC, SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    return {
        "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "linsvm": lambda: LinearSVC(C=1.0),
        "rbfsvm": lambda: SVC(C=2.0, gamma="scale"),
        "randomforest": lambda: RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=0),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                                     early_stopping=True, n_iter_no_change=8, random_state=0),
    }


# parametri skaliranja uče se samo iz trening skupa
def standardize(train, test):
    mu = train.mean(0)
    sd = train.std(0) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


# LOSO izdvaja jednu cijelu sesiju za testiranje
def loso_eval(feats, ys, model_factory=None, dsp_feature_idx=None):
    n = len(feats)
    pooled_true, pooled_pred = [], []
    per_sess = []
    for held in range(n):
        # trening skup čine sve sesije osim izdvojene
        Xtr_parts = []
        ytr_parts = []
        for i in range(n):
            if i != held and len(feats[i]):
                Xtr_parts.append(feats[i])
                ytr_parts.append(np.full(len(feats[i]), ys[i]))
        Xtr = np.vstack(Xtr_parts)
        ytr = np.concatenate(ytr_parts)
        Xte = feats[held]
        if len(Xte) == 0:
            continue
        yte = np.full(len(Xte), ys[held])
        Xtr_s, Xte_s = standardize(Xtr, Xte)
        if dsp_feature_idx is not None:
            # prag jednog obilježja je sredina između prosjeka klasa na trening skupu
            f_tr = Xtr_s[:, dsp_feature_idx]
            thr = 0.5 * (f_tr[ytr == 0].mean() + f_tr[ytr == 1].mean())
            sign = 1 if f_tr[ytr == 1].mean() >= f_tr[ytr == 0].mean() else -1
            pred = ((sign * (Xte_s[:, dsp_feature_idx] - thr)) >= 0).astype(int)
        else:
            clf = model_factory()
            clf.fit(Xtr_s, ytr)
            pred = clf.predict(Xte_s)
        acc = float((pred == yte).mean())
        per_sess.append(acc)
        pooled_true.append(yte)
        pooled_pred.append(pred)
    yt = np.concatenate(pooled_true)
    yp = np.concatenate(pooled_pred)
    pooled_acc = float((yt == yp).mean())
    rec_empty = float((yp[yt == 0] == 0).mean()) if (yt == 0).any() else None
    rec_occ = float((yp[yt == 1] == 1).mean()) if (yt == 1).any() else None
    per_session_round = []
    for a in per_sess:
        per_session_round.append(round(a, 3))
    return {
        "pooled_acc": round(pooled_acc, 4),
        "macro_sess_acc": round(float(np.mean(per_sess)), 4),
        "recall_empty": round(rec_empty, 4) if rec_empty is not None else None,
        "recall_occupied": round(rec_occ, 4) if rec_occ is not None else None,
        "per_session": per_session_round,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="outputs/model_comparison.json")
    ap.add_argument("--max-win", type=int, default=300, help="Max windows per session (uniform downsample) for speed")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # napredak ostaje vidljiv i pri preusmjeravanju izlaza
    except Exception:
        pass

    print("Loading sessions (first run parses + caches; later runs are fast)...")
    sessions = build_sessions(Path(args.data_root))
    common_s = F.common_subcarriers(sessions)
    ys = []
    for s in sessions:
        ys.append(s["y"])
    n_emp = 0
    n_occ = 0
    for s in sessions:
        if s["y"] == 0:
            n_emp += 1
        else:
            n_occ += 1
    print(f"{len(sessions)} sessions  (empty={n_emp}, occupied={n_occ})  common_subcarriers={common_s}")
    for s in sessions:
        print(f"  {s['id']:<48} y={s['y']} fps={s['fps']:.1f} frames={s['amp'].shape[0]}")

    models = make_models()
    results = {}
    for rep in (F.REP_OLD, F.REP_NEW):
        feats = featurize(sessions, rep, common_s, max_win=args.max_win)
        # broj obilježja i ukupan broj prozora
        dim = 0
        for f in feats:
            if len(f):
                dim = f.shape[1]
                break
        nwin = 0
        for f in feats:
            nwin += len(f)
        print(f"\n=== representation {rep.upper()}  feat_dim={dim}  total_windows={nwin} ===")
        results[rep] = {"feat_dim": dim, "total_windows": int(nwin), "models": {}}
        # DSP prag koristi globalnu energiju pokreta za REP_NEW, a globalni std za REP_OLD
        dsp_idx = (2 * common_s + 3) if rep == F.REP_NEW else (2 * common_s + 1)
        r = loso_eval(feats, ys, dsp_feature_idx=dsp_idx)
        results[rep]["models"]["dsp_threshold"] = r
        print(f"  {'dsp_threshold':<14} pooled={r['pooled_acc']:.3f} macro={r['macro_sess_acc']:.3f} "
              f"rec_empty={r['recall_empty']} rec_occ={r['recall_occupied']}")
        for name, factory in models.items():
            r = loso_eval(feats, ys, model_factory=factory)
            results[rep]["models"][name] = r
            print(f"  {name:<14} pooled={r['pooled_acc']:.3f} macro={r['macro_sess_acc']:.3f} "
                  f"rec_empty={r['recall_empty']} rec_occ={r['recall_occupied']}")

    # pobjednik se bira po makro tačnosti sesija zbog neuravnoteženog LOSO skupa
    best = None
    for rep, rd in results.items():
        for name, r in rd["models"].items():
            score = r["macro_sess_acc"]
            if best is None or score > best[2]:
                best = (rep, name, score, r)
    print(f"\n>>> WINNER: representation={best[0]}  model={best[1]}  macro_sess_acc={best[2]:.3f}")
    print(f"    pooled={best[3]['pooled_acc']}  rec_empty={best[3]['recall_empty']}  rec_occ={best[3]['recall_occupied']}")

    sessions_out = []
    for s in sessions:
        sessions_out.append({"id": s["id"], "y": s["y"], "fps": round(s["fps"], 1)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"sessions": sessions_out,
         "common_subcarriers": common_s,
         "winner": {"representation": best[0], "model": best[1], "macro_sess_acc": best[2]},
         "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
