#!/usr/bin/env bash
# Build the stack, create the Groq credential in n8n, import every workflow and publish it.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .env ]] || { echo "Missing .env (copy .env.example and fill it in)" >&2; exit 1; }
set -a; source .env; set +a
[[ -n "${KAGGLE_API_TOKEN:-}" ]] || { echo "KAGGLE_API_TOKEN is empty in .env" >&2; exit 1; }
[[ -n "${GROQ_API_KEY:-}" ]] || { echo "GROQ_API_KEY is empty in .env" >&2; exit 1; }

docker compose up -d --build

echo "Waiting for n8n..."
until curl -sf http://localhost:5678/healthz >/dev/null; do sleep 2; done

# The credential is built from .env inside the container and removed right after import,
# so the key never lands in the repository.
docker compose exec -T -e GROQ_API_KEY n8n sh -c '
  printf "[{\"id\":\"kgGroqCredent01\",\"name\":\"Groq\",\"type\":\"groqApi\",\"data\":{\"apiKey\":\"%s\"}}]" "$GROQ_API_KEY" > /tmp/credentials.json
  n8n import:credentials --input=/tmp/credentials.json
  rm -f /tmp/credentials.json'

docker compose exec -T n8n n8n import:workflow --separate --input=/workflows
for file in n8n/workflows/*.json; do
  id=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$file")
  docker compose exec -T n8n n8n publish:workflow --id="$id"
done

# Published workflows are picked up on start.
docker compose restart n8n
until curl -sf http://localhost:5678/healthz >/dev/null; do sleep 2; done
echo "Ready. n8n: http://localhost:5678 · worker: http://localhost:8000/docs"
