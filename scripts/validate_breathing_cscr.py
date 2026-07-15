import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csi_features as F

# provjera disanja poredi sirovu amplitudu i CSCR na ground-truth sesiji
# konsenzus više podnosilaca smanjuje uticaj lažnih pojedinačnih vrhova
# za svaku fazu traži se vrh od 0.1 do 0.5 Hz i njegova istaknutost
# podnosioci iznad praga daju glas, a konsenzus bpm je medijan njihovih vrhova

PROM_THRESH = 4.0  # najmanja istaknutost vrha koja se računa kao glas


# učitava amplitude, kompleksni CSI, ts_mono i markere faza
def load(jsonl):
    amps, cpx, ts, markers = [], [], [], []
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
                    cpx.append(F.iq_to_complex(v))
                    ts.append(float(r["ts_mono"]))
    if len(amps) < 50:
        return None
    s = min(a.shape[0] for a in amps)
    return (np.stack([a[:s] for a in amps]).astype(np.float32),
            np.stack([c[:s] for c in cpx]).astype(np.complex64),
            np.asarray(ts), markers)


# presemplira na ravnomjernu mrežu jer ts_mono nije ravnomjeran
def resample(mat, ts, fps_u=20.0):
    if ts[-1] - ts[0] < 8:
        return None
    grid = np.arange(ts[0], ts[-1], 1.0 / fps_u)
    out = np.empty((len(grid), mat.shape[1]), np.float32)
    for s in range(mat.shape[1]):
        out[:, s] = np.interp(grid, ts, mat[:, s])
    return out


# medijan vrhova podnosilaca sa izraženim signalom od 0.1 do 0.5 Hz
def consensus_bpm(amp_u, fps_u=20.0):
    T = amp_u.shape[0]
    freqs = np.fft.rfftfreq(T, 1.0 / fps_u)
    band = (freqs >= 0.10) & (freqs <= 0.50)
    if band.sum() < 3:
        return None
    bandf = freqs[band]
    ma = max(2, int(fps_u * 6))
    peak_freqs, proms = [], []
    for s in range(amp_u.shape[1]):
        x = amp_u[:, s].astype(np.float64)
        if x.std() < 1e-6:
            continue
        # klizni prosjek uklanja spori trend i ostavlja bržu komponentu
        c = np.cumsum(np.insert(x, 0, 0.0))
        mov = (c[ma:] - c[:-ma]) / ma
        pad = ma // 2
        movf = np.concatenate([np.full(pad, mov[0]), mov, np.full(len(x) - len(mov) - pad, mov[-1])])[:len(x)]
        P = np.abs(np.fft.rfft((x - movf) * np.hanning(T))) ** 2
        Pb = P[band]
        i = int(np.argmax(Pb))
        prom = Pb[i] / (np.median(Pb) + 1e-12)
        if prom > PROM_THRESH:
            peak_freqs.append(bandf[i])
            proms.append(prom)
    if len(peak_freqs) < 3:
        return {"bpm": None, "n_vote": len(peak_freqs), "spread_bpm": None}
    pf = np.array(peak_freqs)
    return {"bpm": round(float(np.median(pf)) * 60, 1),
            "n_vote": len(pf),
            "spread_bpm": round(float(np.std(pf) * 60), 1)}


def cscr_matrix(cpx, ref):
    denom = cpx[:, ref]
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return np.abs(cpx / denom[:, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--label", default="person_still")
    ap.add_argument("--out", default="outputs/breathing_validation_cscr.json")
    args = ap.parse_args()

    if args.session:
        sdir = Path(args.session)
    else:
        dirs = []
        for p in (Path("data/raw") / args.label).glob("*"):
            if p.is_dir():
                dirs.append(p)
        sdir = max(dirs, key=lambda p: p.stat().st_mtime)
    print(f"session: {sdir}")
    loaded = load(sdir / "records.jsonl")
    if loaded is None:
        print("ERROR: insufficient data")
        return 1
    amp, cpx, ts, markers = loaded
    energy = np.abs(cpx).mean(0)
    ref = int(np.argsort(energy)[len(energy) // 2])
    print(f"frames={amp.shape[0]} subc={amp.shape[1]} dur={ts[-1]-ts[0]:.0f}s cscr_ref_subc={ref}\n")

    # granice faze idu od trenutnog do sljedećeg markera
    bounds = []
    for i in range(len(markers)):
        if i + 1 < len(markers):
            end = markers[i + 1][0]
        else:
            end = ts[-1] + 1
        bounds.append((markers[i][0], end, markers[i][1], markers[i][2]))

    rows = []
    print(f"{'phase':<20}{'target':>7} | {'RAW bpm':>8}{'votes':>6}{'spread':>7} | {'CSCR bpm':>9}{'votes':>6}{'spread':>7}")
    for (t0, t1, phase, tgt) in bounds:
        m = (ts >= t0) & (ts < t1)
        if m.sum() < 80:
            continue
        amp_u = resample(amp[m], ts[m])
        cscr_u = resample(cscr_matrix(cpx[m], ref), ts[m])
        if amp_u is None or cscr_u is None:
            continue
        raw = consensus_bpm(amp_u)
        rat = consensus_bpm(cscr_u)
        rows.append({"phase": phase, "target_bpm": tgt, "raw": raw, "cscr": rat})
        print(f"{str(phase):<20}{str(tgt):>7} | {str(raw['bpm']):>8}{raw['n_vote']:>6}{str(raw['spread_bpm']):>7} | "
              f"{str(rat['bpm']):>9}{rat['n_vote']:>6}{str(rat['spread_bpm']):>7}")

    print("\nTumačenje: dobar kanal u metronomskim fazama daje bpm blizu cilja, više glasova i manji raspon;")
    print("            apneja daje malo nestabilnih glasova, a bolji RAW/CSCR kanal ima zbijeniji konsenzus.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"session": str(sdir), "phases": rows}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
