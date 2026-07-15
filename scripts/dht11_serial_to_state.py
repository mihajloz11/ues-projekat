import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import serial


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "iot" / "latest_state.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# privremeni fajl i preimenovanje sprečavaju djelimično upisan rezultat
def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


# iz serijske linije izdvaja JSON i provjerava da podatak dolazi sa DHT11 senzora
def parse_sensor_line(line):
    start = line.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    if payload.get("sensor") != "dht11":
        return None
    return payload


# dopunjava postojeće stanje sobe temperaturom i vlagom
def merge_dht_state(state, payload, port):
    state = dict(state)
    state["temperature_c"] = payload.get("temperature_c")
    state["humidity_pct"] = payload.get("humidity_pct")
    state["dht11"] = {
        "status": "online",
        "port": port,
        "last_reading_utc": utc_now(),
    }
    state.setdefault("esp32_s3", {})
    state["esp32_s3"].update({
        "role": "receiver_tinyml_and_iot",
        "status": "online",
        "port": port,
    })
    state["ts_utc"] = utc_now()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--duration-sec", type=int, default=0, help="0 means run until stopped")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.duration_sec <= 0:
        deadline = None
    else:
        deadline = time.time() + args.duration_sec
    seen = 0

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        ser.reset_input_buffer()
        while True:
            if deadline is not None and time.time() >= deadline:
                break

            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue

            payload = parse_sensor_line(line)
            if payload is None:
                print(line)
                continue

            state = merge_dht_state(load_state(args.state), payload, args.port)
            save_state(args.state, state)
            seen += 1
            print(json.dumps({
                "temperature_c": state["temperature_c"],
                "humidity_pct": state["humidity_pct"],
                "state_file": str(args.state),
            }))

            if args.once:
                break

    return 0 if seen else 1


if __name__ == "__main__":
    raise SystemExit(main())
