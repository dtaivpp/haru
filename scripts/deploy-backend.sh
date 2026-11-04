#!/usr/bin/env bash
# Deploy the Haru backend to /opt/haru on a Docker + NVIDIA host and (re)start it.
# Usage: scripts/deploy-backend.sh <ssh-host>   (or set HARU_BACKEND_HOST)
set -euo pipefail
HOST="${1:-${HARU_BACKEND_HOST:-}}"
[ -n "$HOST" ] || { echo "usage: $0 <ssh-host>  (or set HARU_BACKEND_HOST)" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST=/opt/haru
for f in gateway.env persona.md; do
  [ -f "$ROOT/backend/config/$f" ] || { echo "backend/config/$f is missing: copy $f.example and fill it in" >&2; exit 1; }
done

ssh "$HOST" "sudo mkdir -p $DEST && sudo chown \$(id -u):\$(id -g) $DEST && mkdir -p $DEST/config $DEST/logs && sudo install -d -o 10001 -g 10001 -m 750 $DEST/data/recordings $DEST/data/conversations $DEST/data/memory $DEST/data/state && sudo install -d -o 1000 -g 1000 $DEST/data/cuda-cache"
rsync -a --delete --exclude secrets.env --exclude overrides.env --exclude google-service-account.json --exclude '*.example' --exclude /data/ --exclude __pycache__ --exclude '*.egg-info' \
  --exclude logs/ --exclude protocol/ "$ROOT/backend/" "$HOST:$DEST/"
rsync -a --delete --exclude __pycache__ --exclude '*.egg-info' "$ROOT/protocol/" "$HOST:$DEST/protocol/"

# First deploy only: mint the terminal token on the server so it never
# passes through this machine's terminal or shell history.
ssh "$HOST" "cd $DEST/config && if [ ! -s secrets.env ]; then
  umask 077; printf 'HARU_TERMINAL_TOKEN=%s\n' \"\$(openssl rand -hex 32)\" > secrets.env; echo 'generated new terminal token'; fi"

# Reproducible image IDs: without these, every build (even fully cached) is
# re-stamped with the current time (image config + provenance attestation), so
# compose sees a "new" image and recreates every container on every deploy.
ssh "$HOST" "cd $DEST && BUILDX_NO_DEFAULT_ATTESTATIONS=1 SOURCE_DATE_EPOCH=0 docker compose up -d --build --remove-orphans
  docker compose ps"
