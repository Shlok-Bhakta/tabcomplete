"""Fake streaming engine for the headless client tests.

POST /v1/predict?stream=1 answers with newline-delimited JSON events chosen
by the request's file name (see SCENARIOS). Prints "PORT <n>" once listening.
Every request and whether the client hung up are appended to the log file
given as the first argument.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[1]
COUNTS = {}


def edit(line, col, text, end_line=None, end_col=None, start_byte=100):
    end_line = line if end_line is None else end_line
    end_col = col if end_col is None else end_col
    return {"start_line": line, "start_col": col, "end_line": end_line, "end_col": end_col,
            "start_byte": start_byte, "end_byte": start_byte + (end_col - col if end_line == line else 7), "text": text}


def final(edits):
    return {"event": "final", "status": "ok" if edits else "empty", "exit_reason": "success" if edits else "no_edits",
            "kind": "local", "edits": edits, "withheld": [], "timings": {"total_ms": 1.0}}


def provisional(text):
    return {"event": "provisional", "edits": [edit(1, 16, text)]}


# Buffer line 2 is "    let x = foo(" with the cursor at its end (col 16).
SCENARIOS = {
    "extend": [(0.0, provisional("ba")), (0.3, provisional("bar")), (0.3, final([edit(1, 16, "bar);")]))],
    "retract": [(0.0, provisional("ba")), (0.4, final([]))],
    "differs": [(0.0, provisional("ba")), (0.4, final([edit(1, 4, "let y", 1, 9, start_byte=90)]))],
    "dismiss": [(0.0, provisional("ba")), (0.4, final([edit(1, 16, "bar);")]))],
    "accept": [(0.0, provisional("ba")), (0.4, final([edit(1, 16, "bar);")]))],
    # First request: slow, should be cancelled by the second (empty) one.
    "cancel#1": [(0.4, provisional("zz")), (0.1, provisional("zzz")), (0.1, final([edit(1, 16, "zzz);")]))],
    "cancel#2": [(0.0, final([]))],
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def log(self, **kw):
        with open(LOG, "a") as f:
            f.write(json.dumps(kw) + "\n")

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.path.startswith("/v1/predict"):
            self.do_GET()
            return
        name = os.path.splitext(os.path.basename(body.get("path", "")))[0]
        COUNTS[name] = COUNTS.get(name, 0) + 1
        key = f"{name}#{COUNTS[name]}" if f"{name}#1" in SCENARIOS else name
        stream = self.path.endswith("?stream=1")
        self.log(key=key, path=self.path, stream=stream)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson" if stream else "application/json")
        self.end_headers()
        if name == "plain":
            # An engine without streaming: one unterminated JSON body.
            out = final([edit(1, 16, "bar);")])
            del out["event"]
            self.wfile.write(json.dumps(out).encode())
            return
        try:
            for delay, ev in SCENARIOS.get(key, [(0.0, final([]))]):
                time.sleep(delay)
                self.wfile.write((json.dumps(ev) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.log(key=key, gone=True)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(f"PORT {server.server_address[1]}", flush=True)
server.serve_forever()
