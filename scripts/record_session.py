import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# vođeno snimanje sesija održava skup podataka dosljednim
# svaki frejm dobija precizno host vrijeme ts_mono, a faze disanja se zapisuju
# kao markeri i u session.json radi označenog ground-truth podatka

# faza sadrži ime, trajanje i ciljani ritam; nula označava zadržavanje daha
PROTOCOLS = {
    # mirna sesija od oko pet minuta sa označenim fazama disanja
    "breathing": [
        ("settle", 20, None),
        ("natural", 70, None),
        ("metronome_12bpm", 60, 12),
        ("metronome_15bpm", 60, 15),
        ("metronome_20bpm", 60, 20),
        ("breath_hold_apnea", 20, 0),
        ("natural_recover", 50, None),
    ],
    # kraća mirna sesija bez metronoma
    "still": [("settle", 15, None), ("still_natural", 285, None)],
    # hodanje sa promjenom putanje i brzine
    "walking": [("walk", 300, None)],
    # prazna soba služi kao referenca
    "empty": [("empty", 300, None)],
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ispisuje uokvirenu poruku i zvučni signal za promjenu faze
def banner(msg):
    line = "=" * max(40, len(msg) + 4)
    print(f"\n{line}\n  {msg}\n{line}\a", flush=True)


def run(args):
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Nedostaje pyserial; instalacija je dostupna kroz scripts\\install_python_tools.cmd") from exc

    # isto parsiranje i putanje kao csi_logger održavaju kompatibilnost zapisa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from csi_logger import build_paths, parse_csi_line

    if args.protocol:
        phases = PROTOCOLS[args.protocol]
    else:
        phases = [("capture", int(args.duration_sec or 300), None)]
    total = 0
    for _, d, _ in phases:
        total += d

    raw_path, jsonl_path, meta_path = build_paths(Path(args.out), args.label, args.session)

    banner(f"PROTOKOL SNIMANJA: {args.protocol or 'plain'}  oznaka={args.label}  ~{total}s")
    print("Provjera uslova snimanja:")
    print("  [ ] obje ESP32 pločice ponovo napajane poslije prethodne sesije")
    print("  [ ] pošiljalac povezan i aktivan na ESP-NOW kanalu")
    print("  [ ] planirano 5-8 sesija po oznaci kroz najmanje tri dana ili položaja")
    if args.protocol == "breathing":
        print("  [ ] metronom spreman za faze od 12, 15 i 20 bpm")
        print("  [ ] osoba miruje, osim prirodnog pomjeranja grudnog koša")
    if args.no_confirm:
        cd = max(0, int(args.countdown))
        for k in range(cd, 0, -1):
            print(f"  početak za {k}s...", flush=True)
            time.sleep(1)
    else:
        try:
            input("\nENTER pokreće snimanje, a Ctrl+C prekida... ")
        except KeyboardInterrupt:
            print("\nSnimanje prekinuto prije početka.")
            return 1

    metadata = {
        "started_utc": utc_now(),
        "port": args.port,
        "baud": args.baud,
        "label": args.label,
        "protocol": args.protocol or "plain",
        "raw_log": str(raw_path),
        "jsonl": str(jsonl_path),
        "notes": "Vođeno snimanje protokola; ts_mono je monotono vrijeme računara, a faze nose ground-truth disanja.",
        "phases": [],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    count = 0
    csi_count = 0
    t0 = time.monotonic()
    phase_idx = -1
    # phase_ends[i] čuva vrijeme završetka faze u sekundama
    phase_ends = []
    acc = 0.0
    for _, dur, _ in phases:
        acc += dur
        phase_ends.append(acc)

    # marker se upisuje u JSONL na početku svake faze
    def write_marker(handle, name, bpm, idx):
        rec = {
            "ts_utc": utc_now(),
            "ts_mono": round(time.monotonic() - t0, 4),
            "line_type": "marker",
            "phase": name,
            "phase_index": idx,
            "target_bpm": bpm,
            "label": args.label,
        }
        handle.write(json.dumps(rec) + "\n")
        handle.flush()

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser, \
                raw_path.open("a", encoding="utf-8", errors="replace") as raw_file, \
                jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            ser.dtr = False
            ser.rts = False
            while True:
                elapsed = time.monotonic() - t0
                if elapsed >= total:
                    break

                # promjena trenutne faze osvježava poruku protokola
                new_idx = len(phases) - 1
                for i, e in enumerate(phase_ends):
                    if elapsed < e:
                        new_idx = i
                        break
                if new_idx != phase_idx:
                    phase_idx = new_idx
                    name, dur, bpm = phases[phase_idx]
                    if bpm == 0:
                        instr = ">>> APNEJA / ZADRŽAVANJE DAHA <<<"
                    elif bpm:
                        instr = f">>> DISANJE UZ METRONOM @ {bpm} bpm <<<"
                    else:
                        instr = f">>> {name.replace('_', ' ').upper()} <<<"
                    banner(f"FAZA {phase_idx + 1}/{len(phases)}: {instr}  (~{dur}s)")
                    metadata["phases"].append(
                        {
                            "name": name,
                            "target_bpm": bpm,
                            "start_mono": round(elapsed, 3),
                            "start_utc": utc_now(),
                        }
                    )
                    write_marker(jsonl_file, name, bpm, phase_idx)

                # poslije prekida serijski port se zatvara i ponovo otvara
                try:
                    data = ser.readline()
                except (serial.SerialException, OSError) as exc:
                    print(f"  [privremena serijska greška; ponovno otvaranje] {exc}", flush=True)
                    try:
                        ser.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    try:
                        ser.open()
                        ser.dtr = False
                        ser.rts = False
                    except Exception as exc2:
                        print(f"  [ponovno otvaranje nije uspjelo; novi pokušaj] {exc2}", flush=True)
                        time.sleep(0.5)
                    continue
                if not data:
                    continue
                line = data.decode(args.encoding, errors="replace").rstrip()
                count += 1
                raw_file.write(line + "\n")

                record = parse_csi_line(line)
                record["ts_mono"] = round(time.monotonic() - t0, 4)  # precizno vrijeme računara
                record["label"] = args.label
                record["port"] = args.port
                record["phase"] = phases[phase_idx][0]
                if record["line_type"] == "csi":
                    csi_count += 1
                jsonl_file.write(json.dumps(record, ensure_ascii=True) + "\n")

                if count % 100 == 0:
                    jsonl_file.flush()
                    raw_file.flush()
                    print(f"  t={elapsed:5.0f}s  lines={count}  csi={csi_count}  phase={phases[phase_idx][0]}")
    except KeyboardInterrupt:
        print("\nSnimanje je prekinuto, a djelimična sesija sačuvana.")
    except serial.SerialException as exc:
        print(f"Greška serijskog porta: {exc}", file=sys.stderr)
        return 2

    # zatvara faze i upisuje završne metapodatke
    for ph in metadata["phases"]:
        ph.setdefault("end_utc", utc_now())
    metadata["ended_utc"] = utc_now()
    metadata["lines"] = count
    metadata["csi_frames"] = csi_count
    eff_fps = round(csi_count / max(1e-6, time.monotonic() - t0), 1)
    metadata["effective_fps"] = eff_fps
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    banner(f"ZAVRŠENO  csi_frames={csi_count}  effective_fps={eff_fps}  -> {jsonl_path.parent}")
    if csi_count < 1000:
        print("UPOZORENJE: mali broj CSI frejmova ukazuje na pošiljaoca, kanal 11 ili vezu pločica.")
    print("Za procjenu modela koriste se sesije iz različitih dana i evaluate_session_holdout.py.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Vođeno snimanje CSI protokola.")
    ap.add_argument("--port", help="Serijski port, na primjer COM3")
    ap.add_argument("--label", help="Oznaka skupa, na primjer person_still / walking / empty_room")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), help="Protokol faza koji ima prednost nad --duration-sec")
    ap.add_argument("--duration-sec", type=float, default=300, help="Trajanje običnog snimanja bez --protocol")
    ap.add_argument("--session", help="Opciona oznaka sesije; podrazumijevano je vrijeme")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--list-protocols", action="store_true")
    ap.add_argument("--no-confirm", action="store_true", help="Preskače ENTER potvrdu kod neinteraktivnog snimanja")
    ap.add_argument("--countdown", type=int, default=0, help="Odbrojavanje u sekundama prije početka snimanja")
    args = ap.parse_args()

    if args.list_protocols:
        print("Dostupni protokoli:")
        for name, phases in PROTOCOLS.items():
            total = 0
            for _, d, _ in phases:
                total += d
            print(f"\n  {name}  (~{total}s)")
            for ph, d, bpm in phases:
                if bpm is None:
                    gt = "free"
                elif bpm == 0:
                    gt = "apnea"
                else:
                    gt = f"{bpm} bpm"
                print(f"    - {ph:<20} {d:>4}s   [{gt}]")
        return 0

    if not args.port or not args.label:
        ap.error("--port i --label su obavezni, osim uz --list-protocols")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
