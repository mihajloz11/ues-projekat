import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csi_features as F

# provjera obrade disanja na snimljenim ground-truth fazama
# markeri phase, target_bpm i ts_mono dijele CSI zapis na segmente
# svaki segment se presemplira ravnomjerno, a FFT i ACF procjenjuju ritam po fazi


# učitava amplitude, ts_mono i markere faza
def load_with_markers(jsonl):
    amps, ts, markers = [], [], []
    with jsonl.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lt = r.get("line_type")
            if lt == "marker":
                markers.append((float(r.get("ts_mono", 0.0)), r.get("phase"), r.get("target_bpm")))
            elif lt == "csi" and "ts_mono" in r:
                v = F.parse_csi_values(r.get("raw", ""))
                if v:
                    amps.append(F.iq_to_amplitude(v))
                    ts.append(float(r["ts_mono"]))
    if len(amps) < 50:
        return None
    s = min(a.shape[0] for a in amps)
    amp = np.stack([a[:s] for a in amps]).astype(np.float32)
    return amp, np.asarray(ts), markers


# presemplira segment na ravnomjernu mrežu
def resample_uniform(seg_amp, seg_ts, fps_u=20.0):
    if seg_ts[-1] - seg_ts[0] < 6:
        return None, fps_u
    grid = np.arange(seg_ts[0], seg_ts[-1], 1.0 / fps_u)
    out = np.empty((len(grid), seg_amp.shape[1]), np.float32)
    for s in range(seg_amp.shape[1]):
        out[:, s] = np.interp(grid, seg_ts, seg_amp[:, s])
    return out, fps_u


# FFT vrh najboljeg podnosioca od 0.1 do 0.5 Hz vraća bpm i SNR
def fft_peak_bpm(amp_u, fps_u):
    T = amp_u.shape[0]
    freqs = np.fft.rfftfreq(T, 1.0 / fps_u)
    band = (freqs >= 0.10) & (freqs <= 0.50)
    ref = (freqs > 0.50) & (freqs <= 1.5)
    if band.sum() < 2 or ref.sum() < 2:
        return None, None
    ma = max(2, int(fps_u * 5))
    best = (0.0, 0.0)  # SNR i bpm
    for s in range(amp_u.shape[1]):
        x = amp_u[:, s].astype(np.float64)
        if x.std() < 1e-6:
            continue
        # spori trend se uklanja prije FFT analize
        c = np.cumsum(np.insert(x, 0, 0.0))
        mov = (c[ma:] - c[:-ma]) / ma
        pad = ma // 2
        movf = np.concatenate([np.full(pad, mov[0]), mov, np.full(len(x) - len(mov) - pad, mov[-1])])[:len(x)]
        xd = x - movf
        P = np.abs(np.fft.rfft(xd * np.hanning(T))) ** 2
        pk = P[band].max()
        snr = pk / (np.median(P[ref]) + 1e-12)
        if snr > best[0]:
            best = (snr, freqs[band][int(np.argmax(P[band]))] * 60.0)
    return round(best[1], 1), round(best[0], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--label", default="person_still")
    ap.add_argument("--out", default="outputs/breathing_validation.json")
    args = ap.parse_args()

    if args.session:
        sdir = Path(args.session)
    else:
        dirs = []
        for p in (Path("data/raw") / args.label).glob("*"):
            if p.is_dir():
                dirs.append(p)
        sdir = max(dirs, key=lambda p: p.stat().st_mtime)
    jsonl = sdir / "records.jsonl"
    print(f"session: {sdir}")
    loaded = load_with_markers(jsonl)
    if loaded is None:
        print("GREŠKA: zapis nema dovoljno CSI/ts_mono podataka za ovu provjeru.")
        return 1
    amp, ts, markers = loaded
    print(f"frames={amp.shape[0]} subc={amp.shape[1]} dur={ts[-1]-ts[0]:.0f}s markers={len(markers)}")
    if not markers:
        print("No phase markers found (need --protocol breathing capture).")
        return 1

    # granice faze idu od trenutnog do sljedećeg markera
    bounds = []
    for i in range(len(markers)):
        if i + 1 < len(markers):
            end = markers[i + 1][0]
        else:
            end = ts[-1] + 1
        bounds.append((markers[i][0], end, markers[i][1], markers[i][2]))

    rows = []
    print(f"\n{'phase':<20}{'target':>8}{'fft_bpm':>9}{'fft_snr':>9}{'acf':>7}{'acf_bpm':>9}{'frames':>8}")
    for (t0, t1, phase, tgt) in bounds:
        m = (ts >= t0) & (ts < t1)
        if m.sum() < 50:
            continue
        amp_u, fps_u = resample_uniform(amp[m], ts[m])
        if amp_u is None:
            continue
        fb, snr = fft_peak_bpm(amp_u, fps_u)
        ranked = F.rank_breathing_subcarriers(amp_u, fps_u, top_k=1)
        if ranked:
            acf, acf_bpm = ranked[0][1], ranked[0][2]
        else:
            acf, acf_bpm = None, None
        rows.append({"phase": phase, "target_bpm": tgt, "fft_bpm": fb, "fft_snr": snr,
                     "acf": round(acf, 3) if acf else None, "acf_bpm": round(acf_bpm, 1) if acf_bpm else None,
                     "frames": int(m.sum())})
        print(f"{str(phase):<20}{str(tgt):>8}{str(fb):>9}{str(snr):>9}{str(round(acf,3) if acf else None):>7}{str(round(acf_bpm,1) if acf_bpm else None):>9}{int(m.sum()):>8}")

    print("\nTumačenje: u metronomskim fazama fft_bpm i acf_bpm prate cilj od 12, 15 ili 20;")
    print("            apneja daje nizak SNR bez stabilnog vrha, a natural stvarni ritam disanja.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"session": str(sdir), "phases": rows}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
