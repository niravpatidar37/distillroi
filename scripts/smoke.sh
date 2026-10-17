#!/usr/bin/env bash
# Smoke test a deployed router: liveness, readiness (student reachable + teacher configured), one classification
# that must return a label from the closed enum. Usage: scripts/smoke.sh <base_url> [id_token]
set -euo pipefail
url="${1:?base url}"
auth=()
if [ -n "${2:-}" ]; then auth=(-H "Authorization: Bearer $2"); fi

curl -fsS --max-time 10 "${auth[@]}" "$url/healthz" >/dev/null
curl -fsS --max-time 10 "${auth[@]}" "$url/readyz"
echo
body=$(curl -fsS --max-time 30 "${auth[@]}" -H 'content-type: application/json' \
  -d '{"text":"My new card still has not arrived, where is it?"}' "$url/v1/classify")
echo "$body"
label=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["label"])')
grep -qx "\"$label\"," src/distill/banking77_labels.json || grep -qx "\"$label\"" src/distill/banking77_labels.json
echo "smoke OK: $label"
