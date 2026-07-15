import argparse
import re
import time

# radar šalje ASCII linije, a "Range NNN" označava grubu udaljenost u centimetrima
RANGE_RE = re.compile(r"^Range\s+(\d+)", re.IGNORECASE)


# iz jedne linije izdvaja stanje radara ili None kada linija nije relevantna
# prihvata sirovu liniju i MMWAVE_TXT oblik koji prosljeđuje firmware
def parse_mmwave_line(line):
    line = line.strip()
    if line.startswith("MMWAVE_TXT,"):
        line = line.split(",", 1)[1].strip()
    if not line:
        return None

    upper = line.upper()
    if upper == "ON":
        return {"present": True}
    if upper == "OFF":
        return {"present": False}
    m = RANGE_RE.match(line)
    if m:
        return {"range_cm": int(m.group(1))}
    return {"raw": line}  # nepoznata linija se zadržava radi novih tipova poruka


# pamti posljednje stanje prisustva i udaljenosti iz toka linija
class MmwaveState:
    def __init__(self, stale_after_s=3.0):
        self.present = None
        self.range_cm = None
        self.last_update = 0.0
        self.stale_after_s = stale_after_s

    def update(self, line):
        parsed = parse_mmwave_line(line)
        if not parsed or "raw" in parsed:
            return False
        if "present" in parsed:
            self.present = parsed["present"]
        if "range_cm" in parsed:
            self.range_cm = parsed["range_cm"]
        self.last_update = time.time()
        return True

    # true kada duže vrijeme nije stigla svježa linija
    def is_stale(self):
        return (time.time() - self.last_update) > self.stale_after_s

    def snapshot(self):
        return {
            "present": self.present,
            "range_cm": self.range_cm,
            "stale": self.is_stale(),
            "age_s": round(time.time() - self.last_update, 2) if self.last_update else None,
        }


def main():
    ap = argparse.ArgumentParser(description="Live mmWave UART monitor via the receiver serial stream.")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    import serial  # lokalni uvoz ostavlja parser upotrebljivim bez pyserial paketa

    state = MmwaveState()
    unknown = []
    count = 0
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        t0 = time.time()
        last_print = 0.0
        while time.time() - t0 < args.seconds:
            line = ser.readline().decode("utf-8", "replace").strip()
            if not line.startswith("MMWAVE_TXT,"):
                continue
            count += 1
            parsed = parse_mmwave_line(line)
            if parsed and "raw" in parsed and parsed["raw"] not in unknown:
                unknown.append(parsed["raw"])
            state.update(line)
            now = time.time()
            if now - last_print >= 0.5:
                snap = state.snapshot()
                print(f"present={snap['present']!s:5s}  range_cm={snap['range_cm']}  age={snap['age_s']}s")
                last_print = now

    print(f"\nmmWave lines seen: {count}")
    if unknown:
        print(f"UNRECOGNIZED message types (extend parser): {unknown}")
    else:
        print("all lines recognized (ON/OFF/Range).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
