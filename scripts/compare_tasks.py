import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csi_features as F
import compare_models as CM

# isti LOSO kao compare_models, razložen na tri zadatka
# presence_all poredi prazno sa bilo kojim zauzetim stanjem
# moving_vs_empty poredi prazno sa hodanjem, a still_vs_empty prazno sa mirovanjem


def make_models_small():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    return {
        "logreg": lambda: LogisticRegression(max_iter=2000),
        "rbfsvm": lambda: SVC(C=2.0, gamma="scale"),
        "randomforest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="outputs/task_decomposition.json")
    ap.add_argument("--max-win", type=int, default=250)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    models = make_models_small()
    out = {}
    for task, occ in CM.TASKS.items():
        sessions = CM.build_sessions(Path(args.data_root), occ_labels=occ)
        common_s = F.common_subcarriers(sessions)
        ys = []
        for s in sessions:
            ys.append(s["y"])
        n_emp = 0
        n_occ = 0
        for y in ys:
            if y == 0:
                n_emp += 1
            else:
                n_occ += 1
        print(f"\n##### TASK {task}  (empty={n_emp}, occupied={n_occ}) #####")
        out[task] = {"n_empty": n_emp, "n_occupied": n_occ, "reps": {}}
        for rep in (F.REP_OLD, F.REP_NEW):
            feats = CM.featurize(sessions, rep, common_s, max_win=args.max_win)
            print(f"  -- {rep} --")
            out[task]["reps"][rep] = {}
            dsp_idx = (2 * common_s + 3) if rep == F.REP_NEW else (2 * common_s + 1)
            r = CM.loso_eval(feats, ys, dsp_feature_idx=dsp_idx)
            out[task]["reps"][rep]["dsp_threshold"] = r
            print(f"    {'dsp_threshold':<13} acc={r['pooled_acc']:.3f} rec_empty={r['recall_empty']} rec_occ={r['recall_occupied']}")
            for name, fac in models.items():
                r = CM.loso_eval(feats, ys, model_factory=fac)
                out[task]["reps"][rep][name] = r
                print(f"    {name:<13} acc={r['pooled_acc']:.3f} rec_empty={r['recall_empty']} rec_occ={r['recall_occupied']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
