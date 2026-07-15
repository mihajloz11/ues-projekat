import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "ruview_bridge" / "static"
LATEST_STATE = PROJECT_ROOT / "iot" / "latest_state.json"
SAMPLE_STATE = PROJECT_ROOT / "iot" / "sample_state.json"


# mali HTTP server servira statičke fajlove i /api/state sa stanjem sobe
class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self.send_state()
            return
        if parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
        super().do_GET()

    # vraća posljednje stanje ili sample dok pravi podatak nije dostupan
    def send_state(self):
        if LATEST_STATE.exists():
            source = LATEST_STATE
        else:
            source = SAMPLE_STATE
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"state": "ERROR", "error": str(exc)}

        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# provjerava postojanje glavnih fajlova i ispravnost sample JSON-a
def validate():
    missing = []
    for path in (STATIC_ROOT / "index.html", SAMPLE_STATE):
        if not path.exists():
            missing.append(path)
    if missing:
        for path in missing:
            print(f"Missing: {path}")
        return 2
    json.loads(SAMPLE_STATE.read_text(encoding="utf-8"))
    print("Demo bridge files are valid.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run the local RuView bridge demo server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--once", action="store_true", help="Validate files and exit.")
    args = parser.parse_args()

    if args.once:
        return validate()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"RuView bridge demo: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
