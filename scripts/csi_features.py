import hashlib
import json
import re
from pathlib import Path

import numpy as np

# zajedničko izdvajanje obilježja iz CSI zapisa za više skripti
# REP_OLD koristi osam frejmova, mean i std po podnosiocu i sedam globalnih statistika
# oslanja se na apsolutnu amplitudu, pa je osjetljiv na drift između sesija
# REP_NEW koristi koeficijent varijacije i energiju pokreta u prozoru od oko 1.5 s
# obilježja su odnosi, pa se zajednički množilac amplitude poništava

CSI_RE = re.compile(r"\[(?P<values>[-0-9,\s]+)\]")
CACHE_DIR = Path("data/cache")


# parsiranje


def parse_csi_values(raw):
    m = CSI_RE.search(raw)
    if not m:
        return None
    out = []
    for part in m.group("values").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            return None
    return out or None


def iq_to_amplitude(values):
    if len(values) % 2:
        values = values[:-1]
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)


# kompleksni CSI po podnosiocu (I + jQ) koristi se za CSCR i fazu
def iq_to_complex(values):
    if len(values) % 2:
        values = values[:-1]
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    return arr[:, 0] + 1j * arr[:, 1]


# local_timestamp sa pločice je 18. polje CSV linije i izražen je u mikrosekundama
def parse_csi_local_ts(raw):
    head = raw.split("[", 1)[0]
    fields = head.split(",")
    if len(fields) > 18:
        try:
            return int(fields[18])
        except ValueError:
            return None
    return None


# učitavanje sesije


# putanja keš fajla izvodi se iz MD5 sažetka putanje sesije
def _cache_path(jsonl_path, kind):
    key = hashlib.md5(str(jsonl_path.resolve()).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{kind}_{key}.npz"


# vraća amplitude, opcioni kompleksni CSI i local_ts jedne sesije
# svi frejmovi se sijeku na najmanji zajednički broj podnosilaca
def load_session(jsonl_path, complex_csi=False, use_cache=True):
    jsonl_path = Path(jsonl_path)
    kind = "cpx" if complex_csi else "amp"
    cache = _cache_path(jsonl_path, kind)
    # važeći keš se koristi kada je noviji od izvornog zapisa
    if use_cache and cache.exists() and cache.stat().st_mtime >= jsonl_path.stat().st_mtime:
        z = np.load(cache, allow_pickle=True)
        out = {"amp": z["amp"], "local_ts": z["local_ts"] if z["local_ts"].size else None}
        if complex_csi:
            out["csi"] = z["csi"]
        return out

    amps, cpxs, tss = [], [], []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("line_type") != "csi":
                continue
            vals = parse_csi_values(rec.get("raw", ""))
            if not vals:
                continue
            amps.append(iq_to_amplitude(vals))
            if complex_csi:
                cpxs.append(iq_to_complex(vals))
            ts = parse_csi_local_ts(rec.get("raw", ""))
            tss.append(ts if ts is not None else -1)
    if len(amps) < 8:
        return None
    s = min(a.shape[0] for a in amps)
    amp = np.stack([a[:s] for a in amps]).astype(np.float32)
    local_ts = np.asarray(tss, dtype=np.int64)
    if (local_ts < 0).any():
        local_ts = np.array([], dtype=np.int64)  # nepotpuno vrijeme se zanemaruje
    out = {"amp": amp, "local_ts": local_ts if local_ts.size else None}
    save = {"amp": amp, "local_ts": local_ts}
    if complex_csi:
        csi = np.stack([c[:s] for c in cpxs]).astype(np.complex64)
        out["csi"] = csi
        save["csi"] = csi
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **save)
    return out


# FPS se prvo procjenjuje iz vremena pločice, a zatim iz trajanja sesije
def effective_fps(session_dir, n_frames, local_ts=None):
    if local_ts is not None and local_ts.size > 10:
        span_us = float(local_ts[-1] - local_ts[0])
        if span_us > 1e6:  # više od jedne sekunde i približno rastuće
            return (local_ts.size - 1) / (span_us / 1e6)
    meta = Path(session_dir) / "session.json"
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
        from datetime import datetime

        dur = (datetime.fromisoformat(m["ended_utc"]) - datetime.fromisoformat(m["started_utc"])).total_seconds()
        if dur > 5:
            return n_frames / dur
    except Exception:
        pass
    return 35.0


# obilježja

REP_OLD = "old"
REP_NEW = "new"


# stara reprezentacija: mean i std po podnosiocu, sedam globalnih mjera i normalizacija prosjekom
def windows_old(amp, window=8, hop=2, normalize=True):
    T, S = amp.shape
    if T < window:
        return np.empty((0, 2 * S + 7), np.float32)
    rows = []
    for start in range(0, T - window + 1, hop):
        ch = amp[start : start + window]
        g = np.array([ch.mean(), ch.std(), ch.min(), ch.max(),
                      np.median(ch), np.percentile(ch, 25), np.percentile(ch, 75)], np.float32)
        v = np.concatenate([ch.mean(0), ch.std(0), g]).astype(np.float32)
        if normalize:
            sc = float(ch.mean())
            if sc > 1e-6:
                v = v / sc
        rows.append(v)
    return np.asarray(rows, np.float32)


# nova reprezentacija koristi duži prozor i otpornija je na drift
# po podnosiocu računa cov = std/mean i motion = mean|delta|/mean
# zajednički množilac amplitude poništava se u odnosima
def windows_new(amp, fps, window_sec=1.5, hop_sec=0.75):
    T, S = amp.shape
    w = max(8, int(round(fps * window_sec)))
    hop = max(1, int(round(fps * hop_sec)))
    if T < w:
        return np.empty((0, 2 * S + 8), np.float32)
    rows = []
    for start in range(0, T - w + 1, hop):
        ch = amp[start : start + w]
        m = ch.mean(0) + 1e-6
        cov = ch.std(0) / m
        dif = np.abs(np.diff(ch, axis=0)).mean(0) / m
        shape = m / m.mean()  # multipath otisak u odnosu na sopstveni nivo
        g = np.array([
            cov.mean(), cov.max(), np.percentile(cov, 90),
            dif.mean(), dif.max(), np.percentile(dif, 90),
            float((cov > 0.05).mean()),          # udio aktivnih podnosilaca
            float(shape.std()),                  # raspršenost normalizovanog otiska
        ], np.float32)
        rows.append(np.concatenate([cov, dif, g]).astype(np.float32))
    return np.asarray(rows, np.float32)


# najmanji broj podnosilaca u svim sesijama održava istu dimenziju obilježja
def common_subcarriers(sessions):
    return min(s["amp"].shape[1] for s in sessions)


# Cross-Subcarrier CSI Ratio (CSCR)


# odnos H(:,m1)/H(:,m2) poništava zajedničku CFO/SFO fazu
# rezultat je čistiji kompleksni signal sa jedne antene
def cscr(csi, m1, m2):
    denom = csi[:, m2]
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return csi[:, m1] / denom


# rangira podnosioce po jačini periodičnosti disanja od 0.1 do 0.5 Hz
# vraća listu (indeks, jačina, bpm) sortiranu od najboljeg rezultata
def rank_breathing_subcarriers(amp, fps, top_k=5):
    T, S = amp.shape
    if T < int(fps * 15):
        return []
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    band = (freqs >= 0.10) & (freqs <= 0.50)
    if band.sum() < 2:
        return []
    lag_min, lag_max = int(round(fps * 2.0)), int(round(fps * 10.0))
    nfft = 1 << (2 * T - 1).bit_length()
    scored = []
    for s in range(S):
        x = amp[:, s].astype(np.float64)
        if x.std() < 1e-6:
            continue
        X = np.fft.rfft(x - x.mean())
        Xb = np.zeros_like(X)
        Xb[band] = X[band]
        xb = np.fft.irfft(Xb, n=T)
        f = np.fft.rfft(xb - xb.mean(), nfft)
        acf = np.fft.irfft(f * np.conj(f), nfft)[:T]
        acf = acf / (acf[0] + 1e-12)
        lo, hi = max(1, lag_min), min(T - 1, lag_max)
        if hi <= lo:
            continue
        seg = acf[lo : hi + 1]
        i = int(np.argmax(seg))
        scored.append((s, float(seg[i]), float(60.0 * fps / (lo + i))))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]
