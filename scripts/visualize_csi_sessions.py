import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSI_RE = re.compile(r"\[(?P<values>[-0-9,\s]+)\]")


# nalazi najnoviji records.jsonl za zadatu oznaku
def latest_records(data_root, label):
    candidates = sorted((data_root / label).glob("*/records.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No records.jsonl found for {label}")
    return candidates[-1]


def parse_values(raw):
    match = CSI_RE.search(raw)
    if not match:
        return None
    values = []
    for part in match.group("values").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def iq_to_amp(values):
    if len(values) % 2:
        values = values[:-1]
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    return np.sqrt(arr[:, 0] * arr[:, 0] + arr[:, 1] * arr[:, 1])


# učitava amplitude sesije do max_frames frejmova
def load_amplitudes(path, max_frames=1200):
    frames = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("line_type") != "csi":
                continue
            values = parse_values(record.get("raw", ""))
            if values:
                frames.append(iq_to_amp(values))
            if len(frames) >= max_frames:
                break
    if not frames:
        raise SystemExit(f"No CSI frames in {path}")
    min_len = min(len(frame) for frame in frames)
    return np.stack([frame[:min_len] for frame in frames])


# klizni prosjek daje mirniju krivu
def rolling_mean(values, window=25):
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


# prosječna amplituda kroz vrijeme za praznu sobu i prisutnu osobu
def save_activity_timeline(empty, person, out):
    empty_trace = rolling_mean(empty.mean(axis=1))
    person_trace = rolling_mean(person.mean(axis=1))
    plt.figure(figsize=(12, 5))
    plt.plot(empty_trace, label="empty_room", linewidth=2)
    plt.plot(person_trace, label="person_present_or_moving", linewidth=2)
    plt.title("CSI average amplitude over time")
    plt.xlabel("Frame window")
    plt.ylabel("Average amplitude")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


# toplotna mapa amplitude po frejmu i podnosiocu
def save_heatmap(data, title, out):
    plt.figure(figsize=(12, 5.2))
    plt.imshow(data.T, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Amplitude")
    plt.title(title)
    plt.xlabel("CSI frame")
    plt.ylabel("Subcarrier/IQ amplitude index")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


# razlika između prisutne osobe i prazne sobe
def save_activity_difference(empty, person, out):
    min_frames = min(len(empty), len(person))
    diff = person[:min_frames].mean(axis=1) - empty[:min_frames].mean(axis=1)
    diff_smooth = rolling_mean(diff)
    plt.figure(figsize=(12, 4.6))
    plt.axhline(0, color="#444444", linewidth=1)
    plt.plot(diff_smooth, color="#c23b22", linewidth=2)
    plt.fill_between(np.arange(len(diff_smooth)), diff_smooth, 0, where=diff_smooth > 0, color="#c23b22", alpha=0.25)
    plt.title("Person session minus empty-room baseline")
    plt.xlabel("Frame window")
    plt.ylabel("Amplitude difference")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


# crtež postavke sobe sa dvije pločice i putanjom kretanja
def save_room_schema(out):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f7f2")
    ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 5, fill=False, linewidth=2.5, color="#222222"))
    ax.scatter([1.5], [3], s=300, marker="s", color="#2f6f9f", label="ESP32 sender")
    ax.scatter([8.5], [3], s=300, marker="s", color="#3a8f5f", label="ESP32-S3 receiver")
    ax.text(1.5, 2.55, "ESP32\nsender", ha="center", va="top", fontsize=11)
    ax.text(8.5, 2.55, "ESP32-S3\nreceiver", ha="center", va="top", fontsize=11)
    ax.plot([1.5, 8.5], [3, 3], color="#666666", linewidth=2, linestyle="--")
    for scale, alpha in [(1.0, 0.18), (1.7, 0.11), (2.4, 0.07)]:
        ellipse = plt.matplotlib.patches.Ellipse(
            (5, 3), width=7.0, height=scale, fill=True, color="#f2b84b", alpha=alpha
        )
        ax.add_patch(ellipse)
    path_x = [4.0, 4.6, 5.2, 5.7, 6.2]
    path_y = [1.6, 2.4, 3.1, 3.8, 4.4]
    ax.plot(path_x, path_y, color="#c23b22", linewidth=3, marker="o", label="primjer kretanja osobe")
    ax.text(5.4, 4.75, "ilustrativna putanja\nnije rekonstruisana", ha="center", fontsize=10, color="#7a1f13")
    ax.text(5, 0.9, "Najkorisnija CSI osjetljivost je uz vezu i okolnu multipath zonu.", ha="center", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Recommended two-node room experiment layout")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create visual plots from recorded CSI sessions.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out-dir", default="outputs/visuals")
    parser.add_argument("--max-frames", type=int, default=1200)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    empty_path = latest_records(data_root, "empty_room")
    person_path = latest_records(data_root, "person_present_or_moving")
    empty = load_amplitudes(empty_path, args.max_frames)
    person = load_amplitudes(person_path, args.max_frames)

    save_activity_timeline(empty, person, out_dir / "csi_activity_timeline.png")
    save_heatmap(empty, "CSI heatmap: empty room", out_dir / "csi_heatmap_empty_room.png")
    save_heatmap(person, "CSI heatmap: person present or moving", out_dir / "csi_heatmap_person_present.png")
    save_activity_difference(empty, person, out_dir / "csi_activity_difference.png")
    save_room_schema(out_dir / "room_experiment_schema.png")

    pngs = []
    for path in out_dir.glob("*.png"):
        pngs.append(str(path))
    summary = {
        "empty_source": str(empty_path),
        "person_source": str(person_path),
        "empty_frames_plotted": int(len(empty)),
        "person_frames_plotted": int(len(person)),
        "outputs": sorted(pngs),
    }
    (out_dir / "visual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
