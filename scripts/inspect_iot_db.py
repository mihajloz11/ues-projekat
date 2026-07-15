import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


# pretvara jedan red baze u rječnik
def row_to_record(row):
    row_id = row[0]
    ts_utc = row[1]
    state = row[2]
    confidence = row[3]
    temperature_c = row[4]
    humidity_pct = row[5]
    mmwave_present = row[6]
    frame_count = row[7]
    csi_status = row[8]
    dht11_status = row[9]
    mmwave_status = row[10]
    payload_json = row[11]

    # JSON payload se pretvara nazad u rječnik
    payload = {}
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            payload = {}

    # CSI dio se izdvaja iz payloada kada postoji
    csi = {}
    if isinstance(payload.get("csi"), dict):
        csi = payload["csi"]

    present = None
    if mmwave_present is not None:
        present = bool(mmwave_present)

    record = {
        "id": row_id,
        "ts_utc": ts_utc,
        "state": state,
        "confidence": confidence,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "mmwave_present": present,
        "frame_count": frame_count,
        "csi_status": csi_status,
        "csi_fps": csi.get("fps"),
        "dht11_status": dht11_status,
        "mmwave_status": mmwave_status,
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Inspect recent room_state rows.")
    parser.add_argument("--db", default="iot/room_state.sqlite")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--minutes", type=float, default=0.0, help="Only include rows newer than this many minutes.")
    parser.add_argument("--json", action="store_true", help="Print JSON for dashboard/API use.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit("SQLite database not found: " + str(db_path))

    # --minutes ograničava upit na novije redove
    where = ""
    params = []
    if args.minutes and args.minutes > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.minutes)
        where = "WHERE ts_utc >= ?"
        params.append(cutoff.isoformat(timespec="seconds"))
    params.append(args.limit)

    # čitaju se posljednji redovi iz baze
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, ts_utc, state, confidence, temperature_c, humidity_pct,
                   mmwave_present, frame_count, csi_status, dht11_status,
                   mmwave_status, payload_json
            FROM room_state
            """
            + where
            + """
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    records = []
    for row in rows:
        records.append(row_to_record(row))

    # --json daje izlaz za dashboard, a bez njega se ispisuje tabela
    if args.json:
        out = {
            "db": str(db_path),
            "minutes": args.minutes,
            "limit": args.limit,
            "count": len(records),
            "rows": records,
        }
        print(json.dumps(out, separators=(",", ":")))
        return 0

    # zaglavlje tabele
    headers = (
        "id",
        "ts_utc",
        "state",
        "conf",
        "temp",
        "hum",
        "mmwave",
        "frames",
        "csi",
        "fps",
        "dht",
        "mm",
    )
    print("\t".join(headers))

    # vrijednosti svakog reda razdvajaju se tabulatorom
    for record in records:
        mmwave = None
        if record["mmwave_present"] is not None:
            mmwave = int(record["mmwave_present"])

        values = (
            record["id"],
            record["ts_utc"],
            record["state"],
            record["confidence"],
            record["temperature_c"],
            record["humidity_pct"],
            mmwave,
            record["frame_count"],
            record["csi_status"],
            record["csi_fps"],
            record["dht11_status"],
            record["mmwave_status"],
        )

        line = []
        for value in values:
            if value is None:
                line.append("")
            else:
                line.append(str(value))
        print("\t".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
