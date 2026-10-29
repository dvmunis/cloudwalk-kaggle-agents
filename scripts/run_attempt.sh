#!/usr/bin/env bash
# Trigger one attempt of the agent team.
#   scripts/run_attempt.sh <competition-slug> [--submit] [--sample-frac 0.2] [--iterations 3]
set -euo pipefail

competition="${1:?usage: run_attempt.sh <competition-slug> [--submit] [--sample-frac F] [--iterations N]}"
shift
submit=false
sample_frac=1
iterations=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) submit=true ;;
    --sample-frac) sample_frac="$2"; shift ;;
    --iterations) iterations="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
  shift
done

body="{\"competition\": \"$competition\", \"submit\": $submit, \"sample_frac\": $sample_frac, \"max_iterations\": $iterations}"

# Right after n8n starts, the webhook takes a few seconds to register (404 until then).
for _ in $(seq 1 15); do
  response=$(curl -sS --max-time 3600 -w '\n%{http_code}' -X POST http://localhost:5678/webhook/kaggle-team \
    -H 'Content-Type: application/json' -d "$body")
  [[ "${response##*$'\n'}" != "404" ]] && break
  sleep 2
done
echo "${response%$'\n'*}"
echo "Trace: curl -s localhost:8000/attempts?competition=$competition"
