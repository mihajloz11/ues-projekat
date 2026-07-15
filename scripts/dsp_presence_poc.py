import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

CSI_RE = re.compile(r"\[(?P<values>[-0-9,\s]+)\]")

# DSP provjera na postojećim CSI zapisima, bez novog snimanja i bez modela
# poredi energiju pokreta za praznu sobu, mirovanje i kretanje
# zatim traži periodični vrh disanja od 0.1 do 0.5 Hz kod mirne osobe
# koristi samo NumPy iz postojećeg virtuelnog okruženja


# parsiranje je usklađeno sa train_baseline_model.py


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


def iq_to_amplitude(values):
    if len(values) % 2:
        values = values[:-1]
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    return np.sqrt((arr[:, 0] * arr[:, 0]) + (arr[:, 1] * arr[:, 1]))


# vraća matricu amplituda sječenu na zajednički broj podnosilaca
def load_amplitude_matrix(jsonl_path):
    frames = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("line_type") != "csi":
                continue
            values = parse_csi_values(record.get("raw", ""))
            if values:
                frames.append(iq_to_amplitude(values))
    if len(frames) < 8:
        return None
    min_len = min(len(f) for f in frames)
    return np.stack([f[:min_len] for f in frames]).astype(np.float32)


# FPS se procjenjuje iz trajanja sesije, uz podrazumijevanu vrijednost 35
def session_fps(session_dir, n_frames):
    meta = session_dir / "session.json"
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
        t0 = datetime.fromisoformat(m["started_utc"])
        t1 = datetime.fromisoformat(m["ended_utc"])
        dur = (t1 - t0).total_seconds()
        if dur > 5:
            return n_frames / dur
    except Exception:
        pass
    return 35.0


# skalar energije pokreta otporan na drift


# u prozoru od oko jedne sekunde računa prosjek std/mean preko podnosilaca
# odnos poništava spori drift amplitude, dok kretanje daje veću varijaciju
def motion_energy_profile(amp, fps):
    win = max(8, int(round(fps * 1.0)))
    hop = max(1, win // 2)
    vals = []
    for start in range(0, amp.shape[0] - win + 1, hop):
        chunk = amp[start : start + win]
        m = chunk.mean(axis=0)
        s = chunk.std(axis=0)
        cov = s / np.maximum(m, 1e-6)  # koeficijent varijacije po podnosiocu
        vals.append(float(np.mean(cov)))
    return np.asarray(vals, dtype=np.float64)


# detektor opsega disanja


def moving_average(x, win):
    if win < 2:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[win:] - c[:-win]) / win
    pad = win // 2
    return np.concatenate([np.full(pad, ma[0]), ma, np.full(len(x) - len(ma) - pad, ma[-1])])[: len(x)]


# najveća normalizovana autokorelacija u zadatom opsegu kašnjenja
# periodično disanje daje izražen vrh; funkcija vraća vrh od 0 do 1 i kašnjenje
def _acf_periodicity(xb, lag_min, lag_max):
    xb = xb - xb.mean()
    n = len(xb)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xb, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n]
    denom = float(acf[0]) + 1e-12
    acf = acf / denom  # acf[0] = 1
    lo = max(1, lag_min)
    hi = min(n - 1, lag_max)
    if hi <= lo:
        return 0.0, 0
    seg = acf[lo : hi + 1]
    i = int(np.argmax(seg))
    return float(seg[i]), lo + i


# periodičnost disanja mjeri se posebno po podnosiocu
# signal se ograničava na 0.1-0.5 Hz, a ACF se mjeri za kašnjenja od 2 do 10 s
# mirna osoba daje jači ACF na više podnosilaca nego prazna soba
def breathing_band_score(amp, fps):
    T = amp.shape[0]
    if T < int(fps * 15):  # najmanje oko 15 s za prepoznavanje periodičnosti disanja
        return None
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    band = (freqs >= 0.10) & (freqs <= 0.50)
    if band.sum() < 2:
        return None
    lag_min = int(round(fps * 2.0))   # 0.50 Hz daje 30 udaha u minuti
    lag_max = int(round(fps * 10.0))  # 0.10 Hz daje 6 udaha u minuti

    best = {"acf": 0.0, "bpm": 0.0, "subc": -1, "concentration": 0.0}
    acfs = []
    for s in range(amp.shape[1]):
        x = amp[:, s].astype(np.float64)
        if x.std() < 1e-6:
            continue
        X = np.fft.rfft(x - x.mean())
        Xband = np.zeros_like(X)
        Xband[band] = X[band]
        xb = np.fft.irfft(Xband, n=T)
        acf_peak, lag = _acf_periodicity(xb, lag_min, lag_max)
        acfs.append(acf_peak)
        band_power = np.abs(X[band]) ** 2
        concentration = float(band_power.max() / (band_power.sum() + 1e-12))
        if acf_peak > best["acf"]:
            best = {
                "acf": acf_peak,
                "bpm": float(60.0 * fps / lag) if lag else 0.0,
                "subc": s,
                "concentration": concentration,
            }
    acfs = np.asarray(acfs)
    best["n_subc_acf_gt_05"] = int((acfs > 0.5).sum())
    best["median_acf"] = float(np.median(acfs)) if acfs.size else 0.0
    return best


# glavni dio


# prolazi kroz sve sesije i računa mjere pokreta i disanja
def analyze(data_root, labels):
    out = {}
    for label in labels:
        for jsonl in sorted((data_root / label).glob("*/records.jsonl")):
            sess_dir = jsonl.parent
            amp = load_amplitude_matrix(jsonl)
            if amp is None:
                continue
            fps = session_fps(sess_dir, amp.shape[0])
            me = motion_energy_profile(amp, fps)
            br = breathing_band_score(amp, fps)
            out.setdefault(label, []).append(
                {
                    "session": sess_dir.name,
                    "frames": int(amp.shape[0]),
                    "subc": int(amp.shape[1]),
                    "fps": round(fps, 1),
                    "motion_p50": round(float(np.median(me)), 4) if me.size else None,
                    "motion_p90": round(float(np.percentile(me, 90)), 4) if me.size else None,
                    "breath_best_acf": round(br["acf"], 3) if br else None,
                    "breath_bpm": round(br["bpm"], 1) if br else None,
                    "breath_subc_acf_gt05": br["n_subc_acf_gt_05"] if br else None,
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument(
        "--labels",
        nargs="*",
        default=["empty_room", "person_still", "walking", "person_present_or_moving"],
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = analyze(Path(args.data_root), args.labels)

    print("\n=== DSP proof-of-concept on existing logs ===")
    print(f"{'label':<26}{'session':<26}{'fps':>5}{'motion_p50':>11}{'motion_p90':>11}{'breath_acf':>11}{'bpm':>7}{'#acf>.5':>9}")
    summ = {}
    for label, rows in results.items():
        m90 = []
        acf = []
        for r in rows:
            print(
                f"{label:<26}{r['session']:<26}{r['fps']:>5}"
                f"{str(r['motion_p50']):>11}{str(r['motion_p90']):>11}"
                f"{str(r['breath_best_acf']):>11}{str(r['breath_bpm']):>7}{str(r['breath_subc_acf_gt05']):>9}"
            )
            if r["motion_p90"] is not None:
                m90.append(r["motion_p90"])
            if r["breath_best_acf"] is not None:
                acf.append(r["breath_best_acf"])
        summ[label] = {
            "n": len(rows),
            "motion_p90_mean": round(float(np.mean(m90)), 4) if m90 else None,
            "breath_acf_mean": round(float(np.mean(acf)), 3) if acf else None,
        }

    print("\n=== sažetak po oznakama (prosjek sesija) ===")
    for label, s in summ.items():
        print(f"  {label:<26} n={s['n']:<3} motion_p90={s['motion_p90_mean']}  breath_acf={s['breath_acf_mean']}")

    print(
        "\nTumačenje: motion_p90 očekivano daje walking >> still ~ empty.\n"
        "            breath_acf očekivano daje person_still >> empty_room.\n"
        "            oba odnosa zajedno potvrđuju razdvajanje pokreta i mirnog prisustva."
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"per_session": results, "summary": summ}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
