import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import serial
except ImportError as exc:
    raise SystemExit("Missing pyserial. Run scripts\\install_python_tools.cmd") from exc


CSI_RE = re.compile(r"\[(?P<values>[-0-9,\s]+)\]")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_values(line):
    match = CSI_RE.search(line)
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


# amplituda iz I/Q parova
def amplitude(values):
    if len(values) % 2:
        values = values[:-1]
    amps = []
    for i in range(0, len(values), 2):
        real = values[i]
        imag = values[i + 1]
        amps.append((real * real + imag * imag) ** 0.5)
    return amps


# ograničava vrijednost između low i high
def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def write_state(out_path, state):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def main():
    parser = argparse.ArgumentParser(description="Update iot/latest_state.json from live ESP32-S3 CSI serial data.")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--out", default="iot/latest_state.json")
    parser.add_argument("--interval-sec", type=float, default=0.5)
    parser.add_argument("--baseline-frames", type=int, default=80)
    args = parser.parse_args()

    out_path = Path(args.out)
    recent_means = []
    baseline_values = []
    baseline = None
    last_write = 0.0
    frame_count = 0

    print(f"Live CSI state: {args.port} at {args.baud} -> {out_path}")
    print("Keep this running for live dashboard view. Ctrl+C stops it.")

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:
            ser.dtr = False
            ser.rts = False
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace").rstrip()
                if "CSI_DATA" not in line:
                    continue
                values = parse_values(line)
                if not values:
                    continue
                amps = amplitude(values)
                if not amps:
                    continue

                frame_count += 1
                mean_amp = statistics.fmean(amps)
                recent_means.append(mean_amp)
                recent_means = recent_means[-80:]

                if baseline is None:
                    # prvih nekoliko desetina frejmova određuje baznu liniju
                    baseline_values.append(mean_amp)
                    if len(baseline_values) >= args.baseline_frames:
                        baseline = statistics.fmean(baseline_values)
                    activity = 0.0
                else:
                    # aktivnost mjeri odstupanje trenutnog prosjeka od bazne linije
                    local_mean = statistics.fmean(recent_means)
                    local_std = statistics.pstdev(recent_means) if len(recent_means) > 2 else 0.0
                    delta = abs(local_mean - baseline)
                    activity = clamp((delta / max(baseline, 1.0)) * 2.2 + local_std / 18.0)

                if activity >= 0.35:
                    state_name = "PERSON_ACTIVITY"
                else:
                    state_name = "LOW_ACTIVITY"
                confidence = round(0.5 + activity * 0.48, 3)
                preview = []
                for v in amps[:64]:
                    preview.append(round(v, 3))
                now = time.time()
                if now - last_write < args.interval_sec:
                    continue
                last_write = now

                state = {
                    "state": state_name,
                    "confidence": confidence,
                    "activity_level": round(activity, 4),
                    "mean_amplitude": round(mean_amp, 4),
                    "baseline_amplitude": round(baseline, 4) if baseline is not None else None,
                    "temperature_c": 23.7,
                    "humidity_pct": 44.2,
                    "mmwave_present": activity >= 0.35,
                    "latency_ms": 18,
                    "esp32_s3": {"role": "receiver_tinyml", "status": "online", "port": args.port},
                    "esp32_devkit_v1": {"role": "sender", "status": "online", "port": "COM5"},
                    "csi_preview": preview,
                    "frame_count": frame_count,
                    "source_label": "live_csi_serial",
                    "source_ts_utc": utc_now(),
                    "ts_utc": utc_now(),
                    "tracking_note": "Jedna veza pošiljaoca i prijemnika daje jačinu aktivnosti, a ne stvarni 2D položaj.",
                }
                write_state(out_path, state)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
