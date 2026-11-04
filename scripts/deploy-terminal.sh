#!/usr/bin/env bash
# Deploy haru-terminal to a Raspberry Pi at ~/haru-terminal and (re)start the service.
# Copies the terminal token straight from the backend host to the Pi over SSH
# pipes, so it never lands in local files or shell history.
# Usage: scripts/deploy-terminal.sh <pi-host> <backend-host>
#        (or set HARU_PI_HOST / HARU_BACKEND_HOST)
set -euo pipefail
PI="${1:-${HARU_PI_HOST:-}}"
BACKEND="${2:-${HARU_BACKEND_HOST:-}}"
[ -n "$PI" ] && [ -n "$BACKEND" ] || { echo "usage: $0 <pi-host> <backend-host>" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST=haru-terminal

ssh "$PI" "mkdir -p $DEST/src $DEST/config"
rsync -a --delete --exclude __pycache__ --exclude '*.egg-info' "$ROOT/protocol/" "$PI:$DEST/src/protocol/"
rsync -a --delete --exclude __pycache__ --exclude '*.egg-info' "$ROOT/terminal/" "$PI:$DEST/src/terminal/"
ssh "$PI" "test -f $DEST/config/config.toml || cp $DEST/src/terminal/config/example.toml $DEST/config/config.toml"

ssh "$BACKEND" "grep '^HARU_TERMINAL_TOKEN=' /opt/haru/config/secrets.env" \
  | ssh "$PI" "umask 077 && cat > $DEST/config/secrets.env"

ssh "$PI" "set -e; cd $DEST
  test -x .venv/bin/python || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade ./src/protocol ./src/terminal
  # The units are templates: fill in this account's user and home directory.
  for u in haru-terminal haru-display; do
    sed \"s#@USER@#\$(id -un)#g; s#@HOME@#\$HOME#g\" src/terminal/systemd/\$u.service | sudo tee /etc/systemd/system/\$u.service >/dev/null
  done
  sudo systemctl daemon-reload
  sudo systemctl enable -q haru-terminal haru-display
  sudo systemctl restart haru-terminal haru-display
  sleep 2; systemctl --no-pager --lines=8 status haru-terminal haru-display"
