import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csi_features as F
from validate_breathing_cscr import consensus_bpm

# provjera mirnog prisustva poredi konsenzus disanja sa praznom sobom
# u prozoru od 60 s računa broj podnosilaca koji se slažu oko vrha od 0.1 do 0.5 Hz
# osoba koja diše daje više glasova i uži raspon nego prazna soba


# klizni prozor vraća rezultate konsenzusa po sesiji
def windows_consensus(amp, fps, win_sec=60, hop_sec=30):
    w = int(fps * win_sec)
    hop = int(fps * hop_sec)
    if amp.shape[0] < w:
        return []
    res = []
    for st in range(0, amp.shape[0] - w + 1, hop):
        # prozor se presemplira na ravnomjernih 20 Hz koje očekuje konsenzus
        seg = amp[st:st + w]
        T = seg.shape[0]
        grid_n = int(win_sec * 20)
        idx = np.linspace(0, T - 1, grid_n).astype(int)
        c = consensus_bpm(seg[idx], fps_u=20.0)
        if c:
            res.append(c)
    return res


# sabira glasove i raspon kroz sve sesije jedne klase
def summarize(label, sessions):
    allv, alls, allbpm = [], [], []
    for sid, amp, fps in sessions:
        rs = windows_consensus(amp, fps)
        v = []
        sp = []
        bp = []
        for r in rs:
            v.append(r["n_vote"])
            if r["spread_bpm"] is not None:
                sp.append(r["spread_bpm"])
            if r["bpm"] is not None:
                bp.append(r["bpm"])
        allv += v
        alls += sp
        allbpm += bp
        print(f"  {sid:<40} windows={len(rs):>2} mean_votes={np.mean(v):.0f} mean_spread={np.mean(sp) if sp else None}")
    print(f"  >>> {label}: mean_votes={np.mean(allv):.0f}  mean_spread={np.mean(alls):.1f}  median_bpm={np.median(allbpm) if allbpm else None}")
    return {"mean_votes": float(np.mean(allv)), "mean_spread": float(np.mean(alls)) if alls else None}


def main():
    # najnovija person_still sesija sadrži ground-truth faze disanja
    still_dirs = []
    for p in Path("data/raw/person_still").glob("*"):
        if p.is_dir():
            still_dirs.append(p)
    still = max(still_dirs, key=lambda p: p.stat().st_mtime)
    still_sess = []
    s = F.load_session(still / "records.jsonl")
    if s:
        fps = F.effective_fps(still, s["amp"].shape[0], s.get("local_ts"))
        still_sess.append((still.name, s["amp"], fps))

    empty_sess = []
    for jsonl in sorted(Path("data/raw/empty_room").glob("*/records.jsonl")):
        s = F.load_session(jsonl)
        if s:
            fps = F.effective_fps(jsonl.parent, s["amp"].shape[0], s.get("local_ts"))
            empty_sess.append((jsonl.parent.name, s["amp"], fps))

    print("STILL (osoba koja diše):")
    still_stat = summarize("STILL", still_sess)
    print("\nEMPTY sesije:")
    empty_stat = summarize("EMPTY", empty_sess)

    print("\n=== razdvajanje ===")
    print(f"  votes: still={still_stat['mean_votes']:.0f} vs empty={empty_stat['mean_votes']:.0f}")
    print(f"  spread: still={still_stat['mean_spread']} vs empty={empty_stat['mean_spread']}")
    print("  Tumačenje: više glasova i manji raspon za still znači da konsenzus disanja")
    print("  razdvaja mirnu osobu od prazne sobe kada mean/variance i mmWave nisu dovoljni.")
    Path("outputs/still_breathing_presence.json").write_text(
        json.dumps({"still": still_stat, "empty": empty_stat}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
