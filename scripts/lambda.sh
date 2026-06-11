#!/usr/bin/env bash
# Lambda Cloud control for the rag-plus-cpt H100 box.
# Key is read from ~/.config/lambda/api_key (chmod 600, NOT in git). Never hardcode it here.
#
# Usage:
#   scripts/lambda.sh ls                 # list instances
#   scripts/lambda.sh up                 # launch H100 + attach persistent FS, print IP
#   scripts/lambda.sh down <instance_id> # terminate (FS persists)
#   scripts/lambda.sh ssh                # ssh into the (single) running instance
#
# Persistent state lives on the 'rag-with-cpt' filesystem and survives terminate.
set -euo pipefail

API=https://cloud.lambda.ai/api/v1
KEY_FILE="${LAMBDA_KEY_FILE:-$HOME/.config/lambda/api_key}"
[ -r "$KEY_FILE" ] || { echo "missing key file: $KEY_FILE" >&2; exit 1; }
KEY="$(tr -d '[:space:]' < "$KEY_FILE")"

# infra constants (from setup 2026-06-11)
REGION=us-southeast-1
ITYPE=gpu_1x_h100_sxm5
SSH_KEY=my-mac
FS=rag-with-cpt
SSH_USER=ubuntu

auth() { curl -s -H "Authorization: Bearer $KEY" "$@"; }

cmd="${1:-ls}"; shift || true
case "$cmd" in
  ls)
    auth "$API/instances" | python3 -m json.tool
    ;;
  up)
    auth "$API/instance-operations/launch" \
      -H 'Content-Type: application/json' \
      -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$ITYPE\",\"ssh_key_names\":[\"$SSH_KEY\"],\"file_system_names\":[\"$FS\"],\"quantity\":1}" \
      | python3 -m json.tool
    echo "# poll 'ls' until status=active, then: scripts/lambda.sh ssh"
    ;;
  down)
    id="${1:?usage: down <instance_id>}"
    auth "$API/instance-operations/terminate" \
      -H 'Content-Type: application/json' \
      -d "{\"instance_ids\":[\"$id\"]}" | python3 -m json.tool
    ;;
  ssh)
    ip="$(auth "$API/instances" | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(d[0]["ip"] if d else "")')"
    [ -n "$ip" ] || { echo "no running instance" >&2; exit 1; }
    exec ssh "$SSH_USER@$ip"
    ;;
  *)
    echo "usage: $0 {ls|up|down <id>|ssh}" >&2; exit 1
    ;;
esac
