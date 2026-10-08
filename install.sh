#!/usr/bin/env bash
# AutoDev Agents — install (or update) and start, in one line:
#   curl -fsSL https://raw.githubusercontent.com/karthikkushi/autodev-agents/main/install.sh | bash
# Run it again any time: it updates, then starts. AUTODEV_DIR picks the folder.
set -euo pipefail
DIR="${AUTODEV_DIR:-$HOME/autodev-agents}"
REPO="https://github.com/karthikkushi/autodev-agents.git"

need() { command -v "$1" >/dev/null || { echo "✗ $1 is required — $2"; exit 1; }; }
need git "install it from https://git-scm.com"
need python3 "install Python 3.10+ from https://www.python.org/downloads"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "✗ Python 3.10+ is required (found $(python3 -V 2>&1))"; exit 1; }

if [ -d "$DIR/.git" ]; then
  echo "→ Updating $DIR"
  git -C "$DIR" pull -q --ff-only
else
  echo "→ Downloading to $DIR"
  git clone -q --depth 1 "$REPO" "$DIR"
fi
cd "$DIR"

[ -x .venv/bin/python ] || python3 -m venv .venv
echo "→ Installing Python packages (the first run takes 5–15 minutes, depending on your internet)"
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  # Piped into bash, stdin is this script — so ask on the terminal itself.
  printf "\nPaste a free Gemini API key from https://aistudio.google.com/apikey (or press Enter to skip): "
  key=""
  { read -rs key </dev/tty; } 2>/dev/null || true
  echo
  if [ -n "$key" ]; then
    sed -i.bak "s|^GOOGLE_API_KEY=.*|GOOGLE_API_KEY=$key|" .env && rm -f .env.bak
    echo "✓ Key saved to $DIR/.env (it never leaves your computer)"
  else
    echo "  Skipped — add keys to $DIR/.env any time (see the comments in it)."
  fi
fi

echo
echo "✓ Starting AutoDev — the dashboard opens at http://localhost:8081/ui"
echo "  Next time, run the same command again, or: cd $DIR && .venv/bin/python main.py"
exec .venv/bin/python main.py
