import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_STATE = {
    "state": "PERSON_PRESENT",
    "confidence": 0.91,
    "temperature_c": 23.7,
    "humidity_pct": 44.2,
    "mmwave_present": True,
    "latency_ms": 18,
    "esp32_s3": {"role": "receiver_tinyml", "status": "online", "port": "COM3"},
    "esp32_devkit_v1": {"role": "sender", "status": "online", "port": "COM5"},
    "csi_preview": [2, 6, 11, 18, 23, 19, 13, 7, 4, 9, 15, 22, 30, 26, 17, 10],
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# vraća posljednji red JSONL fajla kao rječnik
def load_last_jsonl(path):
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    return json.loads(last)


# formira stanje sobe, uz realistične nasumične vrijednosti za sample mod
def state_from_record(record, sample):
    state = dict(DEFAULT_STATE)
    if sample:
        state["confidence"] = round(random.uniform(0.82, 0.96), 2)
        state["latency_ms"] = random.randint(15, 30)
        state["temperature_c"] = round(random.uniform(22.5, 24.5), 1)
        state["humidity_pct"] = round(random.uniform(40.0, 49.0), 1)

    if record:
        label = record.get("label", "")
        if label == "empty_room":
            state["state"] = "EMPTY_ROOM"
        else:
            state["state"] = "PERSON_PRESENT"
        state["confidence"] = 0.75
        state["csi_preview"] = record.get("csi_preview") or state["csi_preview"]
        state["source_label"] = label
        state["source_ts_utc"] = record.get("ts_utc")

    state["ts_utc"] = utc_now()
    return state


def publish_mqtt(state, broker, port, topic):
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise SystemExit("Missing paho-mqtt. Run: scripts\\install_python_tools.cmd") from exc

    client = mqtt.Client()
    client.connect(broker, port, keepalive=30)
    client.publish(topic, json.dumps(state), qos=0, retain=False)
    client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Write and optionally publish the TinyML IoT JSON state.")
    parser.add_argument("--from-jsonl", help="Read the last CSI logger JSONL record.")
    parser.add_argument("--out", default="iot/latest_state.json", help="Fajl koji čita lokalni dashboard")
    parser.add_argument("--sample", action="store_true", help="Pravi realistično probno stanje")
    parser.add_argument("--mqtt", action="store_true", help="Uz upis fajla šalje stanje i na MQTT")
    parser.add_argument("--broker", default="127.0.0.1", help="Adresa MQTT brokera")
    parser.add_argument("--port", type=int, default=1883, help="Port MQTT brokera")
    parser.add_argument("--topic", default="room/occupancy/state", help="MQTT topic.")
    args = parser.parse_args()

    if args.from_jsonl:
        record = load_last_jsonl(Path(args.from_jsonl))
    else:
        record = None
    state = state_from_record(record, args.sample or record is None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.mqtt:
        publish_mqtt(state, args.broker, args.port, args.topic)
        print(f"Published MQTT {args.broker}:{args.port} topic={args.topic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
