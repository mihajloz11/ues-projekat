import argparse
import json
import socket
import sqlite3
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from thingspeak_uploader import build_fields, post_thingspeak, resolve_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPIC = "wifi-csi/room/state"


# skuplja rezultate provjera i broji upozorenja i greške
class CheckReport:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def pass_(self, name, detail):
        print(f"PASS {name}: {detail}")

    def warn(self, name, detail):
        self.warnings += 1
        print(f"WARN {name}: {detail}")

    def fail(self, name, detail):
        self.failures += 1
        print(f"FAIL {name}: {detail}")


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# računa koliko je sekundi prošlo od zadatog vremena
def age_seconds(value):
    ts = parse_ts(value)
    if ts is None:
        return None
    return (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()


# sažeti ispis stanja u jednom redu
def compact_state(state):
    csi = state.get("csi") if isinstance(state.get("csi"), dict) else {}
    edge = state.get("edge_tinyml") if isinstance(state.get("edge_tinyml"), dict) else {}
    return (
        f"state={state.get('state')} conf={state.get('confidence')} "
        f"csi={csi.get('status')} fps={csi.get('fps')} "
        f"edge={edge.get('state')} ts={state.get('ts_utc')}"
    )


# provjerava dostupnost MQTT brokera na TCP nivou
def check_tcp(report, host, port, timeout):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            report.pass_("MQTT broker TCP", f"{host}:{port} is reachable")
            return True
    except OSError as exc:
        report.fail("MQTT broker TCP", f"{host}:{port} is not reachable ({exc})")
        return False


def fetch_json(url, timeout):
    req = urllib.request.Request(url, headers={"cache-control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


# provjerava Node-RED REST endpoint sa stanjem sobe
def check_rest(report, url, timeout, stale_warn_sec):
    try:
        state = fetch_json(url, timeout)
    except Exception as exc:
        report.fail("Node-RED REST", f"{url} failed ({exc})")
        return None

    if state.get("state") in {None, "unknown"}:
        report.fail("Node-RED REST", f"endpoint returned no live state: {state}")
        return state

    state_age = age_seconds(state.get("ts_utc"))
    if state_age is not None and state_age > stale_warn_sec:
        report.warn("Node-RED REST", f"{compact_state(state)} age={state_age:.0f}s")
    else:
        report.pass_("Node-RED REST", compact_state(state))
    return state


# čeka jednu poruku na MQTT temi i vraća None poslije isteka vremena
def wait_mqtt_message(host, port, topic, timeout):
    import paho.mqtt.client as mqtt

    event = threading.Event()
    payload = None

    def on_connect(client, _userdata, _flags, _reason_code, _properties=None):
        client.subscribe(topic)

    def on_message(_client, _userdata, msg):
        nonlocal payload
        try:
            payload = json.loads(msg.payload.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            payload = {"_raw": msg.payload.decode("utf-8", "replace")}
        event.set()

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, keepalive=30)
    client.loop_start()
    try:
        if not event.wait(timeout):
            return None
        return payload
    finally:
        client.loop_stop()
        client.disconnect()


def check_mqtt(report, host, port, topic, timeout, stale_warn_sec):
    try:
        payload = wait_mqtt_message(host, port, topic, timeout)
    except Exception as exc:
        report.fail("MQTT topic", f"{topic} subscribe failed ({exc})")
        return None
    if payload is None:
        report.fail("MQTT topic", f"no message on {topic} within {timeout:.1f}s")
        return None
    if "_raw" in payload:
        report.fail("MQTT topic", f"payload is not JSON: {payload['_raw']}")
        return payload

    msg_age = age_seconds(payload.get("ts_utc"))
    if msg_age is not None and msg_age > stale_warn_sec:
        report.warn("MQTT topic", f"{compact_state(payload)} age={msg_age:.0f}s")
    else:
        report.pass_("MQTT topic", compact_state(payload))
    return payload


# provjerava da baza postoji i sadrži najmanje jedan red
def check_sqlite(report, db_path, stale_warn_sec):
    if not db_path.exists():
        report.fail("SQLite", f"database not found: {db_path}")
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT id, ts_utc, state, confidence, temperature_c, humidity_pct,
                       mmwave_present, frame_count, csi_status, dht11_status,
                       mmwave_status, payload_json
                FROM room_state
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.Error as exc:
        report.fail("SQLite", f"query failed ({exc})")
        return None

    if row is None:
        report.fail("SQLite", "room_state table has no rows")
        return None

    payload = {}
    if row[11]:
        try:
            payload = json.loads(row[11])
        except json.JSONDecodeError:
            payload = {}
    csi = payload.get("csi") if isinstance(payload.get("csi"), dict) else {}
    detail = (
        f"id={row[0]} state={row[2]} conf={row[3]} temp={row[4]} "
        f"hum={row[5]} mmwave={row[6]} csi={row[8]} fps={csi.get('fps')} ts={row[1]}"
    )
    row_age = age_seconds(row[1])
    if row_age is not None and row_age > stale_warn_sec:
        report.warn("SQLite", f"{detail} age={row_age:.0f}s")
    else:
        report.pass_("SQLite", detail)
    if payload:
        return payload
    return {"state": row[2], "confidence": row[3], "ts_utc": row[1]}


# provjerava API sa istorijom posljednjih nekoliko minuta
def check_history(report, url, timeout):
    try:
        history = fetch_json(url, timeout)
    except Exception as exc:
        report.warn("SQLite history API", f"{url} not available yet ({exc})")
        return
    rows = history.get("rows") if isinstance(history.get("rows"), list) else []
    if rows:
        report.pass_("SQLite history API", f"{len(rows)} rows from last {history.get('minutes')} min")
    else:
        report.warn("SQLite history API", "endpoint radi, ali traženi vremenski prozor nema redova")


# šalje jedan red na ThingSpeak radi provjere cloud veze
def check_thingspeak(report, state, api_key, skip, require):
    if skip:
        report.warn("ThingSpeak", "skipped by --skip-thingspeak")
        return
    key = resolve_key(api_key)
    if not key:
        message = "API ključ nije pronađen u argumentu, promjenljivoj okruženja ili lokalnom fajlu"
        if require:
            report.fail("ThingSpeak", message)
        else:
            report.warn("ThingSpeak", message)
        return
    if not state:
        report.fail("ThingSpeak", "no REST/MQTT state available to post")
        return
    fields = build_fields(state)
    try:
        entry = post_thingspeak(key, fields)
    except Exception as exc:
        report.fail("ThingSpeak", f"post failed ({exc})")
        return
    if entry.isdigit() and int(entry) > 0:
        report.pass_("ThingSpeak", f"posted {fields} -> entry #{entry}")
    else:
        report.warn("ThingSpeak", f"cloud returned {entry!r}; likely rate limited or rejected")


def main():
    parser = argparse.ArgumentParser(description="One-pass verification for the WiFi CSI IoT final demo.")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--rest-url", default="http://127.0.0.1:1880/api/room-state")
    parser.add_argument("--history-url", default="http://127.0.0.1:1880/api/room-history")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "iot" / "room_state.sqlite"))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--mqtt-timeout", type=float, default=6.0)
    parser.add_argument("--stale-warn-sec", type=float, default=30.0)
    parser.add_argument("--skip-thingspeak", action="store_true")
    parser.add_argument("--require-thingspeak", action="store_true")
    parser.add_argument("--thingspeak-api-key", default=None)
    args = parser.parse_args()

    report = CheckReport()
    check_tcp(report, args.mqtt_host, args.mqtt_port, args.timeout)
    rest_state = check_rest(report, args.rest_url, args.timeout, args.stale_warn_sec)
    mqtt_state = check_mqtt(
        report,
        args.mqtt_host,
        args.mqtt_port,
        args.topic,
        args.mqtt_timeout,
        args.stale_warn_sec,
    )
    sqlite_state = check_sqlite(report, Path(args.db), args.stale_warn_sec)
    check_history(report, args.history_url, args.timeout)
    state_for_cloud = rest_state or mqtt_state or sqlite_state
    check_thingspeak(
        report,
        state_for_cloud,
        args.thingspeak_api_key,
        args.skip_thingspeak,
        args.require_thingspeak,
    )

    print()
    if report.failures:
        print(f"SUMMARY: {report.failures} failed, {report.warnings} warnings")
        return 1
    print(f"SUMMARY: all required checks passed, {report.warnings} warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
