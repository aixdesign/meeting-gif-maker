#!/usr/bin/env bash
# Launch the Meeting GIF Maker locally.
cd "$(dirname "$0")"

# --- friendly environment checks -------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo ""
  echo "  Python 3 is needed but not installed."
  echo "  Run this once in Terminal, then try again:"
  echo "      xcode-select --install"
  echo ""
  read -r -p "  Press Enter to close…"
  exit 1
fi

# --- first run: create the Python environment ------------------------------
# (imageio-ffmpeg bundles ffmpeg itself, so video analysis is fast on every
#  machine with no extra installs)
if [ ! -d venv ]; then
  echo "First run — setting up (takes a minute or two, needs internet)…"
  python3 -m venv venv
  venv/bin/pip install --quiet --upgrade pip
  venv/bin/pip install --quiet flask opencv-python-headless pillow numpy imageio-ffmpeg || {
    echo "  Setup failed — check your internet connection and try again."
    read -r -p "  Press Enter to close…"
    exit 1
  }
fi
# upgrade path: older setups may miss the bundled ffmpeg package
venv/bin/python -c "import imageio_ffmpeg" 2>/dev/null || \
  venv/bin/pip install --quiet imageio-ffmpeg

# --- free port 5001 if a stale instance is still holding it ----------------
STALE=$(lsof -ti tcp:5001 2>/dev/null)
if [ -n "$STALE" ]; then
  echo "Stopping a previous instance on port 5001…"
  echo "$STALE" | xargs kill -9 2>/dev/null
  sleep 1
fi

# --- open the browser automatically once the server answers ----------------
(
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null http://127.0.0.1:5001/ 2>/dev/null; then
      open "http://127.0.0.1:5001"
      exit 0
    fi
    sleep 0.5
  done
) &

echo ""
echo "  Meeting GIF Maker is running."
echo "  Your browser will open automatically → http://127.0.0.1:5001"
echo "  (Keep this window open while you use the tool. Ctrl-C or close it to stop.)"
echo ""
exec venv/bin/python app.py
