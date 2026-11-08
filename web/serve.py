"""web/serve.py - serves web/ for local testing with caching off, so a reload always gets
the current page, sprites and solver (python's plain http.server lets the browser keep old
copies, which mixes old CSS with new sprite sheets).

    python web/serve.py [port]      (default 8770), then open http://localhost:8770/

or double-click web/serve.cmd: the same, in the background with no window (pythonw), so it
outlives the shell that started it. A second start on a port that already answers does nothing.
"""
import functools
import http.server
import os
import pathlib
import socket
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    # Windows lets a second server bind the same port, and the two then split the requests
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(f"something already answers at http://localhost:{port}/")
            return
    # pythonw has no console: the request log would fail and close every connection unanswered
    if sys.stderr is None:
        sys.stdout = sys.stderr = open(os.devnull, "w")
    handler = functools.partial(NoCacheHandler, directory=str(pathlib.Path(__file__).resolve().parent))
    with http.server.ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"serving web/ at http://localhost:{port}/ (no caching)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
