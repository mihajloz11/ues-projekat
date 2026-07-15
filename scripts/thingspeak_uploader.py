import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

# skripta sluša MQTT stanje sobe i šalje odabrana polja na ThingSpeak
# besplatni nalog prihvata osvježavanje najviše jednom u 15 s, pa je period 16 s
# ključ se traži u --api-key, promjenljivoj okruženja i lokalnim fajlovima
# polja nose prisustvo, pouzdanost, temperaturu, vlagu, mmWave i latenciju

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY_FILES = (
    PROJECT_ROOT / "iot" / "thingspeak_key.local.txt",
    PROJECT_ROOT / "iot" / "thingspeak_key.txt",
)


# traži Write API ključ redom po podržanim izvorima
def resolve_key(cli_key):
    if cli_key:
        return cli_key.strip()
    env = os.environ.get("THINGSPEAK_API_KEY", "").strip()
    if env:
        return env
    for key_file in KEY_FILES:
        if key_file.exists():
            for line in key_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    return ""


# pretvara stanje sobe u ThingSpeak polja
def build_fields(state):
    present = 1 if state.get("state") == "person_present" else 0
    edge = state.get("edge_tinyml") or {}
    return {
        "field1": present,
        "field2": round(float(state.get("confidence") or 0.0), 3),
        "field3": state.get("temperature_c"),
        "field4": state.get("humidity_pct"),
        "field5": 1 if state.get("mmwave_present") else 0,
        "field6": edge.get("latency_us"),
    }


# šalje jedno osvježavanje običnim HTTP GET zahtjevom
def post_thingspeak(api_key, fields, timeout=8.0):
    params = {"api_key": api_key}
    for k, v in fields.items():
        if v is not None:
            params[k] = v
    url = "https://api.thingspeak.com/update?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace").strip()


def main():
    parser = argparse.ArgumentParser(description="MQTT -> ThingSpeak uploader (cloud).")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--topic", default="wifi-csi/room/state")
    parser.add_argument("--period", type=float, default=16.0, help="seconds between cloud posts (>=15)")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = resolve_key(args.api_key)
    if not api_key:
        print("ThingSpeak Write API ključ nije pronađen. Podržani izvori su:")
        for key_file in KEY_FILES:
            print(f"  - {key_file}")
        print("  - THINGSPEAK_API_KEY env var, or --api-key")
        return 2

    import paho.mqtt.client as mqtt

    latest = {}

    # svaka MQTT poruka osvježava posljednje stanje
    def on_message(_c, _u, msg):
        nonlocal latest
        try:
            latest = json.loads(msg.payload.decode("utf-8", "replace"))
        except Exception:
            pass

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()
    client.on_message = on_message
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=30)
    client.subscribe(args.topic)
    client.loop_start()
    print(f"ThingSpeak uploader: {args.topic} -> cloud every {args.period:.0f}s. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(args.period)
            if not latest:
                print("waiting for first MQTT message...")
                continue
            fields = build_fields(latest)
            try:
                entry = post_thingspeak(api_key, fields)
                print(f"posted {fields} -> entry #{entry}")
            except Exception as exc:  # mrežna greška ne prekida dalji rad
                print(f"post failed: {exc}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.loop_stop()
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
