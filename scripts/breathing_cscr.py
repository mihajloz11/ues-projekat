import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csi_features as F

# analiza disanja poredi sirovu amplitudu sa Cross-Subcarrier CSI Ratio (CSCR)
# na postojećim kompleksnim CSI zapisima provjerava da li |H(m)/H(ref)|
# jasnije pokazuje periodičnost disanja od sirove amplitude |H(m)|

SESSIONS = {
    "person_still": 2,             # ciljani slučaj
    "empty_room": 3,               # kontrola sa očekivano niskom vrijednošću
    "person_present_or_moving": 1, # pokret prekriva disanje
}


# |H(:,m)/H(:,ref)| za sve m daje matricu odnosa amplitude (T, S)
def cscr_matrix(csi, ref):
    denom = csi[:, ref]
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    ratio = csi / denom[:, None]
    return np.abs(ratio).astype(np.float32)


# najjača periodičnost u opsegu disanja za zadatu matricu
def best_acf(amp, fps):
    scored = F.rank_breathing_subcarriers(amp, fps, top_k=1)
    if not scored:
        return None
    s, acf, bpm = scored[0]
    return {"acf": round(acf, 3), "bpm": round(bpm, 1), "subc": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="outputs/breathing_cscr.json")
    args = ap.parse_args()

    results = {}
    for label, k in SESSIONS.items():
        for jsonl in sorted((Path(args.data_root) / label).glob("*/records.jsonl"))[:k]:
            s = F.load_session(jsonl, complex_csi=True)
            if s is None:
                continue
            fps = F.effective_fps(jsonl.parent, s["amp"].shape[0], s.get("local_ts"))
            amp = s["amp"]
            csi = s["csi"]
            # referentni podnosilac ima srednju energiju i stabilan signal
            energy = np.abs(csi).mean(0)
            ref = int(np.argsort(energy)[len(energy) // 2])
            cscr_amp = cscr_matrix(csi, ref)

            raw = best_acf(amp, fps)
            rat = best_acf(cscr_amp, fps)
            row = {
                "session": jsonl.parent.name, "fps": round(fps, 1),
                "frames": int(amp.shape[0]), "ref_subc": ref,
                "raw_amp": raw, "cscr": rat,
                "timing": "local_ts" if s.get("local_ts") is not None else "duration_est",
            }
            results.setdefault(label, []).append(row)
            print(f"{label:<26} {row['session']:<22} fps={row['fps']:>5} timing={row['timing']:<13} "
                  f"raw_acf={raw['acf'] if raw else None} cscr_acf={rat['acf'] if rat else None} "
                  f"cscr_bpm={rat['bpm'] if rat else None}")

    print("\nTumačenje: veći CSCR ACF za person_still, uz nizak empty rezultat, pokazuje da")
    print("            odnos podnosilaca izdvaja disanje koje sirova amplituda ne prepoznaje.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
