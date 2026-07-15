import argparse
import json
import math
import re
import sqlite3
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError as exc:
    raise SystemExit("Nedostaje pyserial; instalacija je dostupna kroz scripts\\install_python_tools.cmd") from exc


CSI_RE = re.compile(r"\[(?P<values>[-0-9,\s]+)\]")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# iz CSI linije izdvaja cijele brojeve ili None za neispravnu liniju
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


# senzorska ili TinyML linija sadrži JSON blok koji se vraća kao rječnik
def parse_sensor_line(line):
    line_type = None
    if "SENSOR_DATA" in line:
        line_type = "sensor"
    elif "TINYML_DATA" in line:
        line_type = "tinyml"
    if line_type is None:
        return None
    start = line.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload["_line_type"] = line_type
    return payload


# iz I/Q parova računa amplitudu po podnosiocu
def iq_to_amplitude(values):
    if len(values) % 2:
        values = values[:-1]
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    return np.sqrt(arr[:, 0] * arr[:, 0] + arr[:, 1] * arr[:, 1])


WINDOW_NORM_EPS = 1e-6


# true kada je model treniran sa normalizacijom prosjekom prozora
# stariji modeli bez tog polja koriste apsolutna obilježja
def model_normalizes(model):
    return bool(model) and model.get("feature_norm") == "window_mean"


# formira vektor obilježja iz više frejmova: mean, std i globalna statistika
# redoslijed mora ostati usklađen sa make_windows u Pythonu i build_feature u C-u
def make_feature(frames, normalize=False):
    min_len = min(len(frame) for frame in frames)
    chunk = np.stack([frame[:min_len] for frame in frames])
    mean = chunk.mean(axis=0)
    std = chunk.std(axis=0)
    global_features = np.asarray(
        [
            chunk.mean(),
            chunk.std(),
            chunk.min(),
            chunk.max(),
            np.median(chunk),
            np.percentile(chunk, 25),
            np.percentile(chunk, 75),
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([mean, std, global_features]).astype(np.float32)
    if normalize:
        scale = float(chunk.mean())
        if scale > WINDOW_NORM_EPS:
            vector = (vector / scale).astype(np.float32)
    return vector


# prati brzu promjenu CSI signala u odnosu na sporu baznu liniju sobe
# vrijednost služi prikazu, dok glavna odluka ostaje rezultat modela
def update_signal_filter(
    amp,
    baseline_amp,
    baseline_frames,
    smoothed_activity,
    baseline_alpha,
    activity_smoothing,
    activity_threshold,
    min_baseline_frames,
):
    current = amp.astype(np.float32, copy=False)
    if baseline_amp is None:
        baseline = current.copy()
        raw_activity = 0.0
        baseline_frames = 1
        if smoothed_activity is None:
            smoothed = 0.0
        else:
            smoothed = float(smoothed_activity)
    else:
        n = min(len(current), len(baseline_amp))
        if n <= 0:
            baseline = current.copy()
            raw_activity = 0.0
            baseline_frames = 1
            smoothed = 0.0
        else:
            current_slice = current[:n]
            baseline = baseline_amp[:n].astype(np.float32, copy=True)
            denom = max(float(np.mean(np.abs(baseline))), 1e-6)
            raw_activity = float(np.mean(np.abs(current_slice - baseline)) / denom)
            if smoothed_activity is None:
                previous = raw_activity
            else:
                previous = float(smoothed_activity)
            smoothing = max(0.0, min(1.0, float(activity_smoothing)))
            smoothed = (1.0 - smoothing) * previous + smoothing * raw_activity

            # bazna linija se polako pomjera samo tokom mirovanja ili kalibracije
            quiet_enough = raw_activity < max(activity_threshold * 0.75, 0.01)
            calibrating = baseline_frames < min_baseline_frames
            if quiet_enough or calibrating:
                alpha = max(0.0, min(1.0, float(baseline_alpha)))
                baseline = (1.0 - alpha) * baseline + alpha * current_slice
                baseline_frames += 1

    baseline_ready = baseline_frames >= min_baseline_frames
    if not baseline_ready:
        state = "calibrating"
    elif smoothed >= activity_threshold * 1.8:
        state = "strong_motion"
    elif smoothed >= activity_threshold:
        state = "motion"
    else:
        state = "quiet"

    info = {
        "state": state,
        "mode": "adaptive_csi_baseline",
        "baseline_ready": baseline_ready,
        "baseline_frames": baseline_frames,
        "activity_score": round(float(smoothed), 4),
        "raw_activity_score": round(float(raw_activity), 4),
        "threshold": round(float(activity_threshold), 4),
        "note": "Brza mjera aktivnosti WiFi veze; konačno prisustvo određuje ML model.",
    }
    return info, baseline, baseline_frames, smoothed


def relu(x):
    return np.maximum(x, 0.0)


def softmax(logits):
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / max(float(exp.sum()), 1e-12)


# učitava MLP ili nearest-centroid JSON model i priprema težine
def load_model(path):
    model = json.loads(path.read_text(encoding="utf-8"))
    labels = model["labels"]
    scaler_mean = np.asarray(model["scaler"]["mean"], dtype=np.float32)
    scaler_std = np.asarray(model["scaler"]["std"], dtype=np.float32)
    scaler_std[scaler_std < 1e-6] = 1.0
    loaded = {
        "model_type": model.get("model_type", "amplitude_window_nearest_centroid"),
        "model_path": str(path),
        "labels": labels,
        "window_size": int(model["window_size"]),
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        # način normalizacije usklađuje obilježja sa postupkom treniranja modela
        "feature_norm": model.get("feature_norm", "none"),
    }
    if loaded["model_type"] == "amplitude_window_mlp":
        weights = model["weights"]
        loaded["weights"] = {
            "w1": np.asarray(weights["w1"], dtype=np.float32),
            "b1": np.asarray(weights["b1"], dtype=np.float32),
            "w2": np.asarray(weights["w2"], dtype=np.float32),
            "b2": np.asarray(weights["b2"], dtype=np.float32),
        }
    else:
        centroids = {}
        for label in labels:
            centroids[label] = np.asarray(model["centroids"][label], dtype=np.float32)
        loaded["centroids"] = centroids
    return loaded


# predviđa klasu iz vektora obilježja i vraća klasu, pouzdanost i vjerovatnoće
def predict(feature, model):
    x = (feature - model["scaler_mean"]) / model["scaler_std"]
    if model["model_type"] == "amplitude_window_mlp":
        weights = model["weights"]
        h1 = relu(x @ weights["w1"] + weights["b1"])
        probs_arr = softmax(h1 @ weights["w2"] + weights["b2"])
        probs = {}
        for idx, label in enumerate(model["labels"]):
            probs[label] = float(probs_arr[idx])
        best = max(probs, key=probs.get)
        return best, probs[best], probs

    # nearest-centroid poredi udaljenost do svakog centra
    distances = {}
    for label in model["labels"]:
        distances[label] = float(np.linalg.norm(x - model["centroids"][label]))
    best = min(distances, key=distances.get)
    # manja udaljenost daje veću pouzdanost
    inv = {}
    for label, dist in distances.items():
        inv[label] = math.exp(-dist / 15.0)
    total = sum(inv.values()) or 1.0
    probs = {}
    for label, value in inv.items():
        probs[label] = value / total
    return best, probs[best], probs


# bezbjedan upis koristi privremeni fajl i atomsko preimenovanje
# na Windowsu dashboard može kratko zaključati fajl, pa se upis ponavlja
def write_state(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    last_error = None
    for attempt in range(20):
        tmp = path.with_name(f"{path.name}.{time.time_ns()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
    print(f"Upozorenje: upis stanja je preskočen jer je {path} zaključan: {last_error}")


def append_history(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def load_previous_state(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# provjerava da li je zapamćeno CSI stanje upotrebljivo
def valid_last_known_csi(payload):
    if not isinstance(payload, dict):
        return False
    bad = {None, "no_csi", "sensor_only"}
    return payload.get("state") not in bad and payload.get("zone") not in bad


# pretražuje istoriju unazad i nalazi posljednje važeće CSI stanje
def load_last_known_csi_from_history(path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            recent_lines = deque(handle, maxlen=2000)
    except OSError:
        return None
    for line in reversed(recent_lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("csi", {}).get("status") != "online":
            continue
        candidate = {
            "state": payload.get("state"),
            "zone": payload.get("zone"),
            "raw_prediction": payload.get("raw_prediction"),
            "confidence": payload.get("confidence"),
            "last_frame_utc": payload.get("csi", {}).get("last_frame_utc") or payload.get("source_ts_utc"),
        }
        if valid_last_known_csi(candidate):
            return candidate
    return None


# pravi SQLite tabelu kada ne postoji
def init_sqlite(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS room_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            state TEXT,
            zone TEXT,
            raw_prediction TEXT,
            confidence REAL,
            activity_level REAL,
            mean_amplitude REAL,
            temperature_c REAL,
            humidity_pct REAL,
            mmwave_present INTEGER,
            latency_ms INTEGER,
            frame_count INTEGER,
            csi_status TEXT,
            dht11_status TEXT,
            mmwave_status TEXT,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def append_sqlite(conn, payload):
    if conn is None:
        return
    conn.execute(
        """
        INSERT INTO room_state (
            ts_utc, state, zone, raw_prediction, confidence, activity_level, mean_amplitude,
            temperature_c, humidity_pct, mmwave_present, latency_ms, frame_count,
            csi_status, dht11_status, mmwave_status, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("ts_utc"),
            payload.get("state"),
            payload.get("zone"),
            payload.get("raw_prediction"),
            payload.get("confidence"),
            payload.get("activity_level"),
            payload.get("mean_amplitude"),
            payload.get("temperature_c"),
            payload.get("humidity_pct"),
            1 if payload.get("mmwave_present") else 0,
            payload.get("latency_ms"),
            payload.get("frame_count"),
            payload.get("csi", {}).get("status"),
            payload.get("dht11", {}).get("status"),
            payload.get("mmwave", {}).get("status"),
            json.dumps(payload, separators=(",", ":")),
        ),
    )
    conn.commit()


# šalje stanje na MQTT broker i nastavlja offline kada broker nije dostupan
class MqttPublisher:
    def __init__(self, host, port, topic):
        self.host = host
        self.port = port
        self.topic = topic
        self.status = "disabled"
        self.error = None
        self.client = None
        if not host:
            return
        self._connect()

    def _connect(self):
        try:
            import paho.mqtt.client as mqtt

            try:
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            except AttributeError:
                self.client = mqtt.Client()
            self.client.connect(self.host, self.port, keepalive=30)
            self.client.loop_start()
            self.status = "online"
        except Exception as exc:
            self.client = None
            self.status = "offline"
            self.error = str(exc)

    def publish(self, payload):
        if self.client is None and self.host:
            self._connect()
        if self.client is None:
            return
        try:
            self.client.publish(self.topic, json.dumps(payload, separators=(",", ":")), qos=0, retain=True)
            self.status = "online"
            self.error = None
        except Exception as exc:
            self.status = "offline"
            self.error = str(exc)

    def close(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()


# računa starost ISO vremenske oznake u sekundama
def age_seconds(iso_value, now=None):
    if not iso_value:
        return None
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if now is None:
        now = datetime.now(timezone.utc)
    return max(0.0, (now - parsed).total_seconds())


# svodi različite nazive na tri standardna stanja
def normalize_presence_state(state):
    if state in {"person_present", "present", "PERSON_PRESENT"}:
        return "person_present"
    if state in {"empty_room", "empty", "EMPTY_ROOM"}:
        return "empty_room"
    return "uncertain"


# stabilizuje izlaz modela glasanjem i histerezom
class PresenceDecisionFilter:
    def __init__(self, initial_state, window_size, present_votes, empty_votes, confidence_threshold, mmwave_mode="advisory"):
        self.state = normalize_presence_state(initial_state)
        self.votes = deque(maxlen=max(1, window_size))
        self.present_votes = max(1, present_votes)
        self.empty_votes = max(1, empty_votes)
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        # advisory zadržava CSI kao glavni izvor i samo prijavljuje neslaganje sa mmWave
        # veto pri neslaganju spušta stanje na uncertain
        if mmwave_mode in ("advisory", "veto"):
            self.mmwave_mode = mmwave_mode
        else:
            self.mmwave_mode = "advisory"

    def update(self, model_state, model_confidence, mmwave_present, mmwave_online):
        raw_state = normalize_presence_state(model_state)
        confidence = max(0.0, min(1.0, float(model_confidence)))
        # niska pouzdanost računa se kao uncertain glas
        if confidence >= self.confidence_threshold:
            vote = raw_state
        else:
            vote = "uncertain"
        self.votes.append(vote)
        counts = Counter(self.votes)

        if counts["person_present"] >= self.present_votes:
            self.state = "person_present"
        elif counts["empty_room"] >= self.empty_votes:
            self.state = "empty_room"
        elif self.state == "uncertain":
            self.state = "uncertain"

        output_state = self.state
        warnings = []
        conflict = False
        if mmwave_online:
            disagree = None
            if self.state == "empty_room" and mmwave_present:
                disagree = "CSI daje praznu sobu, a mmWave registruje pokret"
            elif self.state == "person_present" and not mmwave_present:
                disagree = "CSI daje prisustvo, a mmWave ne registruje pokret"
            if disagree:
                conflict = True
                if self.mmwave_mode == "veto":
                    warnings.append(disagree + "; stanje se spušta na uncertain.")
                else:
                    warnings.append(disagree + " (advisory; CSI ostaje glavni izvor).")

        if conflict and self.mmwave_mode == "veto":
            output_state = "uncertain"
            output_confidence = min(confidence, 0.55)
        elif output_state == "uncertain":
            output_confidence = min(confidence, 0.60)
        else:
            output_confidence = confidence

        return {
            "state": output_state,
            "stable_state": self.state,
            "raw_model_state": raw_state,
            "raw_model_confidence": round(confidence, 4),
            "confidence": round(output_confidence, 4),
            "vote_window": list(self.votes),
            "vote_counts": dict(counts),
            "confidence_threshold": self.confidence_threshold,
            "mmwave_used": bool(mmwave_online),
            "mmwave_present": bool(mmwave_present) if mmwave_online else None,
            "conflict": conflict,
            "mmwave_mode": self.mmwave_mode,
            "warnings": warnings,
            "note": "Glavno stanje stabilizuju glasanje i histereza; mmWave je pomoćni izvor osim u veto modu.",
        }


ZONE_LABELS = {"zone_door_left", "zone_middle", "zone_bed_right"}


# razlika između dvije najbolje zone pokazuje jasnoću odluke
def zone_probability_margin(zone, zone_probs):
    if zone not in ZONE_LABELS:
        return 0.0
    zone_score = float(zone_probs.get(zone, 0.0))
    competitors = []
    for label, score in zone_probs.items():
        if label in ZONE_LABELS and label != zone:
            competitors.append(float(score))
    if competitors:
        second_best = max(competitors)
    else:
        second_best = 0.0
    return zone_score - second_best


# zona se prikazuje tek poslije provjere pouzdanosti, margine i kratke stabilnosti
class ZoneDecisionFilter:
    def __init__(self, window_size, min_votes, confidence_threshold, margin_threshold):
        self.votes = deque(maxlen=max(1, window_size))
        self.min_votes = max(1, min_votes)
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.margin_threshold = max(0.0, margin_threshold)

    def update(self, raw_zone, raw_confidence, zone_probs, presence_state):
        if raw_zone not in ZONE_LABELS:
            raw_zone = "zone_uncertain"
        confidence = max(0.0, min(1.0, float(raw_confidence)))
        margin = zone_probability_margin(raw_zone, zone_probs)

        if presence_state != "person_present":
            candidate = "zone_uncertain"
            reason = "presence_not_confirmed"
        elif raw_zone not in ZONE_LABELS:
            candidate = "zone_uncertain"
            reason = "not_a_zone"
        elif confidence < self.confidence_threshold:
            candidate = "zone_uncertain"
            reason = "low_confidence"
        elif margin < self.margin_threshold:
            candidate = "zone_uncertain"
            reason = "weak_margin"
        else:
            candidate = raw_zone
            reason = "accepted_candidate"

        self.votes.append(candidate)
        counts = Counter(self.votes)
        accepted_counts = {}
        for label in ZONE_LABELS:
            accepted_counts[label] = counts[label]
        best_zone, best_count = max(accepted_counts.items(), key=lambda item: item[1])

        if best_count >= self.min_votes and candidate == best_zone:
            output_zone = best_zone
            status = "accepted"
        elif best_count >= self.min_votes:
            output_zone = "zone_uncertain"
            status = "holding"
        else:
            output_zone = "zone_uncertain"
            if any(accepted_counts.values()):
                status = "warming"
            else:
                status = "uncertain"

        return {
            "zone": output_zone,
            "status": status,
            "raw_zone": raw_zone,
            "raw_confidence": round(confidence, 4),
            "margin": round(float(margin), 4),
            "candidate": candidate,
            "reason": reason,
            "vote_window": list(self.votes),
            "vote_counts": dict(counts),
            "min_votes": self.min_votes,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "note": "Gruba zona se prikazuje tek poslije provjere pouzdanosti, margine i kratke stabilnosti.",
        }


# sastavlja payload za JSON, dashboard i bazu
def make_payload(
    previous,
    state,
    zone,
    raw_prediction,
    confidence,
    zone_probabilities,
    activity_level,
    mean_amplitude,
    latest_temperature,
    latest_humidity,
    latest_mmwave_present,
    dht11_status,
    dht11_error,
    dht11_fail_count,
    dht11_gpio,
    dht11_last_reading,
    mmwave_status,
    mmwave_last_reading,
    args,
    frame_count,
    csi_preview,
    csi_last_frame_utc,
    csi_fps_estimate,
    edge_tinyml,
    signal_filter,
    mqtt_status,
    mqtt_error,
    decision,
    note,
):
    ts_utc = utc_now()
    now = datetime.now(timezone.utc)
    csi_age = age_seconds(csi_last_frame_utc, now)
    dht_age = age_seconds(dht11_last_reading, now)
    mmwave_age = age_seconds(mmwave_last_reading, now)
    csi_status = "online" if csi_age is not None and csi_age <= 5 else "offline"
    if csi_fps_estimate is None:
        csi_fps_estimate = previous.get("csi", {}).get("fps")
    try:
        csi_fps_float = float(csi_fps_estimate) if csi_fps_estimate is not None else None
    except (TypeError, ValueError):
        csi_fps_float = None
    max_window_size = max(
        int(getattr(args, "model_window_size", 0) or 0),
        int(getattr(args, "presence_model_window_size", 0) or 0),
    )
    if csi_fps_float and csi_fps_float > 0:
        window_fill_sec = max_window_size / csi_fps_float
    else:
        window_fill_sec = None
    if dht_age is not None and dht_age <= args.dht_stale_sec:
        dht_status_display = "online" if dht11_status == "online" else "last_valid"
    else:
        dht_status_display = "stale"
    mmwave_status_display = "online" if mmwave_age is not None and mmwave_age <= args.mmwave_stale_sec else "stale"
    stale_warnings = []
    if csi_status != "online":
        stale_warnings.append("CSI frame_count se ne mijenja; pošiljalac nije aktivan ili je izvan dometa.")
    elif csi_fps_float is not None and csi_fps_float < 2.0:
        stale_warnings.append("CSI frekvencija je niska, pa dashboard kasni sa osvježavanjem.")
    if dht_status_display == "last_valid":
        stale_warnings.append("DHT11 očitanje nije uspjelo; prikazana je posljednja važeća vrijednost.")
    elif dht_status_display != "online":
        stale_warnings.append("DHT11 koristi posljednju važeću vrijednost.")
    if mmwave_status_display != "online":
        stale_warnings.append("mmWave stanje je zastarjelo; mogući uzrok je veza senzora ili serijski izlaz.")
    if isinstance(decision, dict):
        stale_warnings.extend(decision.get("warnings", []))

    reported_state = state
    reported_zone = zone
    reported_raw_prediction = raw_prediction
    reported_confidence = confidence
    last_known_csi = None
    # kada CSI nije online, prijavljuje se no_csi i čuva posljednje važeće stanje
    if csi_status != "online":
        previous_last_known_csi = previous.get("last_known_csi")
        if valid_last_known_csi(previous_last_known_csi):
            last_known_csi = previous_last_known_csi
        elif state not in {"no_csi", "sensor_only"} and zone not in {"no_csi", "sensor_only"}:
            if isinstance(confidence, (int, float)):
                conf_value = round(float(confidence), 4)
            else:
                conf_value = confidence
            last_known_csi = {
                "state": state,
                "zone": zone,
                "raw_prediction": raw_prediction,
                "confidence": conf_value,
                "last_frame_utc": csi_last_frame_utc,
            }
        reported_state = "no_csi"
        reported_zone = "no_csi"
        reported_raw_prediction = "stale_csi"
        reported_confidence = 0.0

    sender_port = "external_sender" if csi_status == "online" else "not_detected"

    if isinstance(reported_confidence, (int, float)):
        reported_confidence_out = round(float(reported_confidence), 4)
    else:
        reported_confidence_out = reported_confidence
    if isinstance(activity_level, (int, float)):
        activity_level_out = round(float(activity_level), 4)
    else:
        activity_level_out = activity_level

    if edge_tinyml is not None:
        edge_tinyml_out = edge_tinyml
    else:
        edge_tinyml_out = previous.get("edge_tinyml")
    if signal_filter is not None:
        signal_filter_out = signal_filter
    else:
        signal_filter_out = previous.get("signal_filter")
    if decision is not None:
        decision_out = decision
    else:
        decision_out = previous.get("decision")
    if csi_preview is not None:
        csi_preview_out = csi_preview
    else:
        csi_preview_out = previous.get("csi_preview", [])

    return {
        "state": reported_state,
        "zone": reported_zone,
        "raw_prediction": reported_raw_prediction,
        "confidence": reported_confidence_out,
        "last_known_csi": last_known_csi,
        "zone_probabilities": zone_probabilities,
        "activity_level": activity_level_out,
        "mean_amplitude": mean_amplitude,
        "temperature_c": latest_temperature,
        "humidity_pct": latest_humidity,
        "mmwave_present": latest_mmwave_present,
        "edge_tinyml": edge_tinyml_out,
        "signal_filter": signal_filter_out,
        "decision": decision_out,
        "zone_decision": previous.get("zone_decision"),
        "latency_ms": 18,
        "esp32_s3": {"role": "receiver_tinyml_and_iot", "status": "online", "port": args.port},
        "esp32_devkit_v1": {
            "role": "sender",
            "status": csi_status,
            "port": sender_port,
        },
        "csi": {
            "status": csi_status,
            "frame_count": frame_count,
            "last_frame_utc": csi_last_frame_utc,
            "age_sec": round(csi_age, 1) if csi_age is not None else None,
            "fps": round(csi_fps_float, 2) if csi_fps_float is not None else None,
            "window_fill_sec": round(window_fill_sec, 1) if window_fill_sec is not None else None,
        },
        "dht11": {
            "status": dht_status_display,
            "raw_status": dht11_status,
            "port": args.port,
            "gpio": dht11_gpio,
            "error": dht11_error,
            "fail_count": dht11_fail_count,
            "last_reading_utc": dht11_last_reading,
            "age_sec": round(dht_age, 1) if dht_age is not None else None,
        },
        "mmwave": {
            "status": mmwave_status_display,
            "source": "OT2",
            "gpio": 5,
            "last_reading_utc": mmwave_last_reading,
            "age_sec": round(mmwave_age, 1) if mmwave_age is not None else None,
        },
        "mqtt": {"status": mqtt_status, "topic": args.mqtt_topic, "error": mqtt_error},
        "database": {"status": "online" if args.sqlite else "disabled", "path": args.sqlite or None},
        "warnings": stale_warnings,
        "csi_preview": csi_preview_out,
        "frame_count": frame_count,
        "source_label": reported_zone,
        "source_ts_utc": csi_last_frame_utc or previous.get("source_ts_utc"),
        "ts_utc": ts_utc,
        "zone_status": previous.get("zone_status"),
        "tracking_note": note,
        "model": {
            "type": getattr(args, "model_type", None),
            "path": args.model,
            "window_size": getattr(args, "model_window_size", None),
            "presence_type": getattr(args, "presence_model_type", None),
            "presence_path": args.presence_model,
            "presence_window_size": getattr(args, "presence_model_window_size", None),
            "zone_confidence_threshold": args.zone_confidence_threshold,
            "zone_min_margin": args.zone_min_margin,
            "zone_stable_window": args.zone_stable_window,
            "zone_stable_votes": args.zone_stable_votes,
            "zone_output_mode": args.zone_output_mode,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Pokreće CSI inferenciju uživo i objavljuje stanje prostorije.")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--model", default="models/zone_csi_centroid.json")
    parser.add_argument("--presence-model", default="", help="Opcioni binarni model prisustva za glavno stanje")
    parser.add_argument("--motion-model", default="", help="Opcioni still-vs-moving model za stanje uz prisutnu osobu")
    parser.add_argument("--motion-stable-window", type=int, default=5, help="Prozor većinskog glasanja za still/moving stanje")
    parser.add_argument("--zone-confidence-threshold", type=float, default=0.90)
    parser.add_argument("--zone-min-margin", type=float, default=0.25)
    parser.add_argument("--zone-stable-window", type=int, default=9)
    parser.add_argument("--zone-stable-votes", type=int, default=6)
    parser.add_argument(
        "--zone-output-mode",
        choices=("conservative", "off"),
        default="conservative",
        help="Vrijednost off zadržava grube zone samo kao dijagnostiku",
    )
    parser.add_argument("--out", default="iot/latest_state.json")
    parser.add_argument("--smooth", type=int, default=3, help="Broj posljednjih predikcija za većinsko glasanje")
    parser.add_argument("--decision-window", type=int, default=7)
    parser.add_argument("--decision-present-votes", type=int, default=3)
    parser.add_argument("--decision-empty-votes", type=int, default=5)
    parser.add_argument("--decision-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--mmwave-mode", choices=["advisory", "veto"], default="advisory",
                        help="advisory zadržava CSI kao glavni izvor; veto može spustiti stanje na uncertain")
    parser.add_argument("--write-every", type=float, default=0.75)
    parser.add_argument("--signal-baseline-alpha", type=float, default=0.02)
    parser.add_argument("--signal-activity-smoothing", type=float, default=0.35)
    parser.add_argument("--signal-activity-threshold", type=float, default=0.08)
    parser.add_argument("--signal-min-baseline-frames", type=int, default=16)
    parser.add_argument("--dht-stale-sec", type=float, default=90.0)
    parser.add_argument("--mmwave-stale-sec", type=float, default=90.0)
    parser.add_argument("--history", default="iot/state_history.jsonl")
    parser.add_argument("--history-every", type=float, default=5.0)
    parser.add_argument("--sqlite", default="iot/room_state.sqlite")
    parser.add_argument("--sqlite-every", type=float, default=5.0)
    parser.add_argument("--mqtt-host", default="")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", default="wifi-csi/room/state")
    parser.add_argument("--mqtt-every", type=float, default=2.0)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="Nula označava rad do ručnog prekida")
    args = parser.parse_args()

    out_path = Path(args.out)
    sqlite_conn = init_sqlite(Path(args.sqlite)) if args.sqlite else None
    mqtt_publisher = MqttPublisher(args.mqtt_host, args.mqtt_port, args.mqtt_topic)
    zone_model = load_model(Path(args.model))
    presence_model = load_model(Path(args.presence_model)) if args.presence_model else None
    motion_model = load_model(Path(args.motion_model)) if args.motion_model else None
    max_window_size = max(
        zone_model["window_size"],
        presence_model["window_size"] if presence_model else 0,
        motion_model["window_size"] if motion_model else 0,
    )
    args.model_type = zone_model["model_type"]
    args.model_window_size = zone_model["window_size"]
    args.presence_model_type = presence_model["model_type"] if presence_model else None
    args.presence_model_window_size = presence_model["window_size"] if presence_model else None
    frames = deque(maxlen=max_window_size)
    votes = deque(maxlen=args.smooth)
    motion_votes = deque(maxlen=max(1, args.motion_stable_window))
    last_write = 0.0
    last_history = 0.0
    last_sqlite = 0.0
    last_mqtt = 0.0
    frame_count = 0
    previous = load_previous_state(out_path)
    if not valid_last_known_csi(previous.get("last_known_csi")) and args.history:
        recovered_last_known = load_last_known_csi_from_history(Path(args.history))
        if recovered_last_known:
            previous["last_known_csi"] = recovered_last_known
    previous_frame_count = previous.get("frame_count")
    if isinstance(previous_frame_count, int):
        frame_count = previous_frame_count
    if isinstance(previous.get("decision"), dict):
        previous_decision = previous.get("decision")
    else:
        previous_decision = {}
    decision_filter = PresenceDecisionFilter(
        previous_decision.get("stable_state") or previous.get("state"),
        window_size=args.decision_window,
        present_votes=args.decision_present_votes,
        empty_votes=args.decision_empty_votes,
        confidence_threshold=args.decision_confidence_threshold,
        mmwave_mode=args.mmwave_mode,
    )
    zone_filter = ZoneDecisionFilter(
        window_size=args.zone_stable_window,
        min_votes=args.zone_stable_votes,
        confidence_threshold=args.zone_confidence_threshold,
        margin_threshold=args.zone_min_margin,
    )
    latest_temperature = previous.get("temperature_c")
    latest_humidity = previous.get("humidity_pct")
    dht11_status = previous.get("dht11", {}).get("status", "unknown")
    dht11_error = previous.get("dht11", {}).get("error")
    dht11_fail_count = previous.get("dht11", {}).get("fail_count")
    dht11_gpio = previous.get("dht11", {}).get("gpio")
    dht11_last_reading = previous.get("dht11", {}).get("last_reading_utc")
    latest_mmwave_present = bool(previous.get("mmwave_present", False))
    mmwave_status = previous.get("mmwave", {}).get("status", "unknown")
    mmwave_last_reading = previous.get("mmwave", {}).get("last_reading_utc")
    if isinstance(previous.get("edge_tinyml"), dict):
        latest_edge_tinyml = previous.get("edge_tinyml")
    else:
        latest_edge_tinyml = None
    if isinstance(previous.get("signal_filter"), dict):
        latest_signal_filter = previous.get("signal_filter")
    else:
        latest_signal_filter = None
    csi_last_frame_utc = previous.get("csi", {}).get("last_frame_utc") or previous.get("source_ts_utc")
    csi_fps_estimate = previous.get("csi", {}).get("fps")
    csi_frame_times = deque(maxlen=80)
    baseline_amp = None
    baseline_frames = 0
    smoothed_activity = None
    started = time.time()

    print(f"Inferencija zone uživo na {args.port}, model {args.model}")
    if presence_model:
        print(f"Model prisustva: {args.presence_model}")
    print(f"Objedinjeno CSI/DHT11 stanje se upisuje u {out_path}")
    print("Ctrl+C stops it.")

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:
            ser.dtr = False
            ser.rts = False
            while True:
                if args.duration_sec and time.time() - started >= args.duration_sec:
                    break
                data = ser.readline()
                if not data:
                    # bez nove linije povremeno se upisuju postojeći senzorski podaci
                    now = time.time()
                    if now - last_write >= args.write_every:
                        payload = make_payload(
                            previous=previous,
                            state=previous.get("state", "no_csi"),
                            zone=previous.get("zone", "no_csi"),
                            raw_prediction=previous.get("raw_prediction", "no_csi"),
                            confidence=previous.get("confidence", 0),
                            zone_probabilities=previous.get("zone_probabilities", {}),
                            activity_level=previous.get("activity_level", 0),
                            mean_amplitude=previous.get("mean_amplitude"),
                            latest_temperature=latest_temperature,
                            latest_humidity=latest_humidity,
                            latest_mmwave_present=latest_mmwave_present,
                            dht11_status=dht11_status,
                            dht11_error=dht11_error,
                            dht11_fail_count=dht11_fail_count,
                            dht11_gpio=dht11_gpio,
                            dht11_last_reading=dht11_last_reading,
                            mmwave_status=mmwave_status,
                            mmwave_last_reading=mmwave_last_reading,
                            args=args,
                            frame_count=frame_count,
                            csi_preview=previous.get("csi_preview", []),
                            csi_last_frame_utc=csi_last_frame_utc,
                            csi_fps_estimate=csi_fps_estimate,
                            edge_tinyml=latest_edge_tinyml,
                            signal_filter=latest_signal_filter,
                            mqtt_status=mqtt_publisher.status,
                            mqtt_error=mqtt_publisher.error,
                            decision=previous.get("decision") if isinstance(previous.get("decision"), dict) else None,
                            note="Nema svježe serijske linije; posljednje CSI stanje ostaje zastarjelo do novih ESP32-S3 podataka.",
                        )
                        write_state(out_path, payload)
                        if args.history and now - last_history >= args.history_every:
                            append_history(Path(args.history), payload)
                            last_history = now
                        if now - last_sqlite >= args.sqlite_every:
                            append_sqlite(sqlite_conn, payload)
                            last_sqlite = now
                        if now - last_mqtt >= args.mqtt_every:
                            mqtt_publisher.publish(payload)
                            last_mqtt = now
                        last_write = now
                        previous = payload
                    continue
                line = data.decode("utf-8", "replace").rstrip()

                sensor_payload = parse_sensor_line(line)
                if sensor_payload is not None:
                    line_type = sensor_payload.pop("_line_type", "sensor")
                    if line_type == "tinyml":
                        latest_edge_tinyml = dict(sensor_payload)
                        if "person_probability_milli" in latest_edge_tinyml:
                            latest_edge_tinyml["person_probability"] = round(
                                float(latest_edge_tinyml["person_probability_milli"]) / 1000.0, 4
                            )
                        if "confidence_milli" in latest_edge_tinyml:
                            latest_edge_tinyml["confidence"] = round(
                                float(latest_edge_tinyml["confidence_milli"]) / 1000.0, 4
                            )
                        latest_edge_tinyml["last_reading_utc"] = utc_now()
                    elif sensor_payload.get("sensor") == "dht11":
                        if "temperature_c" in sensor_payload:
                            latest_temperature = sensor_payload["temperature_c"]
                        if "humidity_pct" in sensor_payload:
                            latest_humidity = sensor_payload["humidity_pct"]
                        dht11_status = sensor_payload.get("status", "online")
                        dht11_error = sensor_payload.get("error")
                        dht11_fail_count = sensor_payload.get("fail_count")
                        dht11_gpio = sensor_payload.get("gpio", dht11_gpio)
                        if dht11_status == "online":
                            dht11_last_reading = utc_now()
                    elif sensor_payload.get("sensor") == "mmwave":
                        latest_mmwave_present = bool(sensor_payload.get("present", False))
                        mmwave_status = "online"
                        mmwave_last_reading = utc_now()

                    now = time.time()
                    if now - last_write >= args.write_every:
                        sensor_payload_out = dict(previous)
                        sensor_payload_out = make_payload(
                            previous=sensor_payload_out,
                            state=sensor_payload_out.get("state", "sensor_only"),
                            zone=sensor_payload_out.get("zone", "sensor_only"),
                            raw_prediction=sensor_payload_out.get("raw_prediction", "sensor_only"),
                            confidence=sensor_payload_out.get("confidence", 0),
                            zone_probabilities=sensor_payload_out.get("zone_probabilities", {}),
                            activity_level=sensor_payload_out.get("activity_level", 0),
                            mean_amplitude=sensor_payload_out.get("mean_amplitude"),
                            latest_temperature=latest_temperature,
                            latest_humidity=latest_humidity,
                            latest_mmwave_present=latest_mmwave_present,
                            dht11_status=dht11_status,
                            dht11_error=dht11_error,
                            dht11_fail_count=dht11_fail_count,
                            dht11_gpio=dht11_gpio,
                            dht11_last_reading=dht11_last_reading,
                            mmwave_status=mmwave_status,
                            mmwave_last_reading=mmwave_last_reading,
                            args=args,
                            frame_count=frame_count,
                            csi_preview=sensor_payload_out.get("csi_preview", []),
                            csi_last_frame_utc=csi_last_frame_utc,
                            csi_fps_estimate=csi_fps_estimate,
                            edge_tinyml=latest_edge_tinyml,
                            signal_filter=latest_signal_filter,
                            mqtt_status=mqtt_publisher.status,
                            mqtt_error=mqtt_publisher.error,
                            decision=previous.get("decision") if isinstance(previous.get("decision"), dict) else None,
                            note="Osvježeni su samo senzori; za CSI podatke mora biti aktivan ESP32 pošiljalac.",
                        )
                        write_state(out_path, sensor_payload_out)
                        if args.history and now - last_history >= args.history_every:
                            append_history(Path(args.history), sensor_payload_out)
                            last_history = now
                        if now - last_sqlite >= args.sqlite_every:
                            append_sqlite(sqlite_conn, sensor_payload_out)
                            last_sqlite = now
                        if now - last_mqtt >= args.mqtt_every:
                            mqtt_publisher.publish(sensor_payload_out)
                            last_mqtt = now
                        last_write = now
                        previous = sensor_payload_out
                    continue

                if "CSI_DATA" not in line:
                    continue
                values = parse_values(line)
                if not values:
                    continue
                amp = iq_to_amplitude(values)
                latest_signal_filter, baseline_amp, baseline_frames, smoothed_activity = update_signal_filter(
                    amp,
                    baseline_amp,
                    baseline_frames,
                    smoothed_activity,
                    baseline_alpha=args.signal_baseline_alpha,
                    activity_smoothing=args.signal_activity_smoothing,
                    activity_threshold=args.signal_activity_threshold,
                    min_baseline_frames=args.signal_min_baseline_frames,
                )
                frames.append(amp)
                frame_count += 1
                csi_frame_times.append(time.time())
                if len(csi_frame_times) >= 2:
                    elapsed = csi_frame_times[-1] - csi_frame_times[0]
                    if elapsed > 0:
                        csi_fps_estimate = (len(csi_frame_times) - 1) / elapsed
                csi_last_frame_utc = utc_now()
                if len(frames) < max_window_size:
                    continue

                zone_feature = make_feature(list(frames)[-zone_model["window_size"]:], normalize=model_normalizes(zone_model))
                zone_predicted, zone_confidence, zone_probs = predict(zone_feature, zone_model)
                votes.append(zone_predicted)
                smoothed_zone = Counter(votes).most_common(1)[0][0]
                smoothed_zone_conf = zone_probs.get(smoothed_zone, zone_confidence)

                presence_probability = None
                presence_state = smoothed_zone
                presence_confidence = smoothed_zone_conf
                presence_probs = {}
                if presence_model is not None:
                    presence_feature = make_feature(list(frames)[-presence_model["window_size"]:], normalize=model_normalizes(presence_model))
                    presence_predicted, presence_confidence_raw, presence_probs = predict(presence_feature, presence_model)
                    presence_probability = float(
                        presence_probs.get("person_present", 1.0 - presence_probs.get("empty_room", 0.0))
                    )
                    if presence_probability >= 0.5:
                        presence_state = "person_present"
                    else:
                        presence_state = "empty_room"
                    if presence_state == "person_present":
                        presence_confidence = presence_probability
                    else:
                        presence_confidence = 1.0 - presence_probability

                motion_raw = None
                motion_state = None
                motion_conf = None
                motion_probs = {}
                if motion_model is not None:
                    motion_feature = make_feature(list(frames)[-motion_model["window_size"]:], normalize=model_normalizes(motion_model))
                    motion_raw, motion_conf, motion_probs = predict(motion_feature, motion_model)
                    motion_votes.append(motion_raw)
                    motion_state = Counter(motion_votes).most_common(1)[0][0]

                zone_decision = zone_filter.update(
                    raw_zone=smoothed_zone,
                    raw_confidence=smoothed_zone_conf,
                    zone_probs=zone_probs,
                    presence_state=presence_state,
                )
                if args.zone_output_mode == "off":
                    zone_decision = dict(zone_decision)
                    zone_decision["zone"] = "zone_uncertain"
                    zone_decision["status"] = "disabled"
                    zone_decision["note"] = "Gruba zona je isključena na dashboardu; raw_zone ostaje samo dijagnostički podatak."

                now = time.time()
                if now - last_write < args.write_every:
                    continue
                last_write = now

                mmwave_age_now = age_seconds(mmwave_last_reading)
                mmwave_online_now = mmwave_age_now is not None and mmwave_age_now <= args.mmwave_stale_sec
                decision = decision_filter.update(
                    model_state=presence_state,
                    model_confidence=presence_confidence,
                    mmwave_present=latest_mmwave_present,
                    mmwave_online=mmwave_online_now,
                )
                reported_state = decision["state"]
                if reported_state == "empty_room":
                    reported_zone = "empty_room"
                elif reported_state == "person_present":
                    reported_zone = zone_decision["zone"]
                else:
                    reported_zone = "zone_uncertain"

                # vjerovatnoće zona zaokružuju se za prikaz
                zone_probabilities = {}
                for k, v in zone_probs.items():
                    zone_probabilities[k] = round(float(v), 4)
                if presence_probability is not None:
                    activity_level = presence_probability
                else:
                    activity_level = 1.0 - float(zone_probs.get("empty_room", 0.0))
                csi_preview = []
                for v in amp[:64].tolist():
                    csi_preview.append(round(float(v), 3))

                payload = make_payload(
                    previous=previous,
                    state=reported_state,
                    zone=reported_zone,
                    raw_prediction=zone_predicted,
                    confidence=decision["confidence"],
                    zone_probabilities=zone_probabilities,
                    activity_level=activity_level,
                    mean_amplitude=round(float(amp.mean()), 4),
                    latest_temperature=latest_temperature,
                    latest_humidity=latest_humidity,
                    latest_mmwave_present=latest_mmwave_present,
                    dht11_status=dht11_status,
                    dht11_error=dht11_error,
                    dht11_fail_count=dht11_fail_count,
                    dht11_gpio=dht11_gpio,
                    dht11_last_reading=dht11_last_reading,
                    mmwave_status=mmwave_status,
                    mmwave_last_reading=mmwave_last_reading,
                    args=args,
                    frame_count=frame_count,
                    csi_preview=csi_preview,
                    csi_last_frame_utc=csi_last_frame_utc,
                    csi_fps_estimate=csi_fps_estimate,
                    edge_tinyml=latest_edge_tinyml,
                    signal_filter=latest_signal_filter,
                    mqtt_status=mqtt_publisher.status,
                    mqtt_error=mqtt_publisher.error,
                    decision=decision,
                    note="Prisustvo je glavni rezultat modela; zona je gruba eksperimentalna procjena jedne WiFi CSI veze.",
                )
                presence_prob_out = {}
                for k, v in presence_probs.items():
                    presence_prob_out[k] = round(float(v), 4)
                payload["presence"] = {
                    "state": presence_state,
                    "confidence": round(float(presence_confidence), 4),
                    "person_probability": (
                        round(float(presence_probability), 4) if presence_probability is not None else None
                    ),
                    "probabilities": presence_prob_out,
                }
                if motion_model is not None:
                    motion_map = {"person_still": "still", "walking": "moving"}
                    # still/moving se prikazuje kada soba nije pouzdano prazna
                    # uncertain često označava prisutnu osobu u pokretu
                    present_now = reported_state != "empty_room"
                    motion_prob_out = {}
                    for k, v in motion_probs.items():
                        motion_prob_out[motion_map.get(k, k)] = round(float(v), 4)
                    if present_now:
                        motion_state_out = motion_map.get(motion_state, motion_state)
                    else:
                        motion_state_out = "n_a"
                    payload["motion"] = {
                        "state": motion_state_out,
                        "raw_state": motion_map.get(motion_raw, motion_raw),
                        "confidence": round(float(motion_conf), 4) if motion_conf is not None else None,
                        "probabilities": motion_prob_out,
                        "enabled": True,
                        "active": bool(present_now),
                        "note": "Still-vs-moving podmodel ima smisla samo kada je osoba prisutna.",
                    }
                payload["zone_decision"] = zone_decision
                payload["zone_confidence"] = round(float(smoothed_zone_conf), 4)
                if reported_zone in ZONE_LABELS:
                    payload["zone_status"] = zone_decision["status"]
                else:
                    payload["zone_status"] = reported_zone
                write_state(out_path, payload)
                if args.history and now - last_history >= args.history_every:
                    append_history(Path(args.history), payload)
                    last_history = now
                if now - last_sqlite >= args.sqlite_every:
                    append_sqlite(sqlite_conn, payload)
                    last_sqlite = now
                if now - last_mqtt >= args.mqtt_every:
                    mqtt_publisher.publish(payload)
                    last_mqtt = now
                previous = payload
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        mqtt_publisher.close()
        if sqlite_conn is not None:
            sqlite_conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
