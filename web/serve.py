"""web/serve.py - serves web/ for local testing with caching off, so a reload always gets
the current page, sprites and solver (python's plain http.server lets the browser keep old
copies, which mixes old CSS with new sprite sheets).

    python web/serve.py [port]      (default 8770), then open http://localhost:8770/
"""
import functools
import http.server
import pathlib
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    handler = functools.partial(NoCacheHandler, directory=str(pathlib.Path(__file__).resolve().parent))
    with http.server.ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"serving web/ at http://localhost:{port}/ (no caching)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
