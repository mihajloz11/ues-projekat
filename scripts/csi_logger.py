import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import serial
except ImportError as exc:
    raise SystemExit("Missing pyserial. Run: scripts\\install_python_tools.cmd") from exc


# iz CSI linije izdvaja brojeve između uglastih zagrada
CSI_RE = re.compile(r"\[(?P<values>[-0-9,\s]+)\]")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# pretvara jednu serijsku liniju u rječnik pogodan za JSON zapis
def parse_csi_line(line):
    # vrsta linije određuje način parsiranja
    line_type = "log"
    if "SENSOR_DATA" in line or "MMWAVE_TXT" in line:
        line_type = "sensor"
    elif "CSI_DATA" in line:
        line_type = "csi"

    record = {
        "ts_utc": utc_now(),
        "raw": line,
        "line_type": line_type,
        "csi_len": 0,
        "csi_preview": [],
    }

    # senzorska linija već sadrži JSON blok
    if line_type == "sensor":
        start = line.find("{")
        if start >= 0:
            try:
                record["sensor"] = json.loads(line[start:])
            except json.JSONDecodeError:
                pass
        return record

    match = CSI_RE.search(line)
    if not match:
        return record

    # brojevi se izdvajaju redom iz teksta
    values = []
    for part in match.group("values").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue

    record["csi_len"] = len(values)
    record["csi_preview"] = values[:64]
    if values:
        # prosjek apsolutnih vrijednosti daje grubu jačinu signala
        total = 0
        for v in values:
            total += abs(v)
        record["amp_mean_abs"] = round(total / len(values), 4)
    return record


# priprema putanje za raw log, JSONL i metapodatke
def build_paths(out_dir, label, session):
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")
    if not safe_label:
        safe_label = "unlabeled"
    session_id = session or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / safe_label / session_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "raw_serial.log", base / "records.jsonl", base / "session.json"


def main():
    parser = argparse.ArgumentParser(description="Log ESP32/ESP32-S3 CSI serial output.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    parser.add_argument("--label", required=True, help="Dataset label, for example empty_room.")
    parser.add_argument("--session", help="Optional session id. Defaults to timestamp.")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate.")
    parser.add_argument("--out", default="data/raw", help="Output directory.")
    parser.add_argument("--max-lines", type=int, default=0, help="Stop after this many lines; 0 means unlimited.")
    parser.add_argument("--duration-sec", type=float, default=0, help="Stop after seconds; 0 means unlimited.")
    parser.add_argument("--encoding", default="utf-8", help="Serial text encoding.")
    args = parser.parse_args()

    raw_path, jsonl_path, meta_path = build_paths(Path(args.out), args.label, args.session)
    started = time.time()
    count = 0

    metadata = {
        "started_utc": utc_now(),
        "port": args.port,
        "baud": args.baud,
        "label": args.label,
        "raw_log": str(raw_path),
        "jsonl": str(jsonl_path),
        "notes": "CSI receiver session. Keep hardware layout and room state stable during capture.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Logging {args.port} at {args.baud} baud")
    print(f"Raw log:   {raw_path}")
    print(f"Records:   {jsonl_path}")
    print("Press Ctrl+C to stop.")

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser, raw_path.open(
            "a", encoding="utf-8", errors="replace"
        ) as raw_file, jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            # isključeni DTR/RTS sprečavaju reset pločice pri otvaranju porta
            ser.dtr = False
            ser.rts = False
            while True:
                if args.duration_sec and time.time() - started >= args.duration_sec:
                    break
                if args.max_lines and count >= args.max_lines:
                    break

                data = ser.readline()
                if not data:
                    continue
                line = data.decode(args.encoding, errors="replace").rstrip()
                count += 1
                raw_file.write(line + "\n")
                raw_file.flush()

                record = parse_csi_line(line)
                record["label"] = args.label
                record["port"] = args.port
                jsonl_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                jsonl_file.flush()

                # rjeđi ispis ne usporava čitanje sa porta
                if count % 50 == 0:
                    print(f"{count:06d} {record['line_type']} len={record['csi_len']}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 2

    metadata["ended_utc"] = utc_now()
    metadata["lines"] = count
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Lines captured: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
