"""Static cache builder for GitHub Pages deployment.

GitHub Pages only serves static files (no Python backend), so instead of
running server.py's HTTP server, a scheduled GitHub Action runs this script
to fetch fresh data and write cache.json directly into the repo. The
frontend (app.js) then reads that static cache.json instead of calling a
live /api/watchlist endpoint.

Run:
    python build_cache.py
"""

import json
from pathlib import Path

from server import build_watchlist_payload

CACHE_FILE = Path(__file__).parent / "cache.json"

if __name__ == "__main__":
    payload = build_watchlist_payload()
    CACHE_FILE.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {CACHE_FILE} with {len(payload['stocks'])} stocks.")
