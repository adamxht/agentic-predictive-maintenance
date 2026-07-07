#!/bin/sh
# One-shot: registers ClickHouse as OpenLIT's "Default DB" via its own web
# API, working around an upstream bug where prisma/seed.js -- the script
# that's supposed to do this from the INIT_DB_* env vars -- crashes before
# reaching that step on every container start (see the comment above the
# openlit service in docker-compose.agent-tracing.yml for the full story).
# Idempotent: skips if a "Default DB" config is already registered, so
# re-running this on every `compose up` is harmless.
set -eu

COOKIE_JAR=/tmp/openlit-cookies.txt

echo "Waiting for OpenLIT UI at $OPENLIT_URL..."
until curl -sf "$OPENLIT_URL/api/auth/providers" >/dev/null 2>&1; do
  sleep 2
done

# The account itself is seeded fine (the crash happens later, in the
# db-config step) -- log in as it to drive the same UI API a human would
# use under Settings -> Database Config.
CSRF_TOKEN=$(curl -s -c "$COOKIE_JAR" "$OPENLIT_URL/api/auth/csrf" \
  | sed -n 's/.*"csrfToken":"\([^"]*\)".*/\1/p')

curl -s -b "$COOKIE_JAR" -c "$COOKIE_JAR" -X POST "$OPENLIT_URL/api/auth/callback/login" \
  --data-urlencode "csrfToken=$CSRF_TOKEN" \
  --data-urlencode "email=$OPENLIT_SEED_EMAIL" \
  --data-urlencode "password=$OPENLIT_SEED_PASSWORD" \
  --data-urlencode "json=true" >/dev/null

EXISTING_CONFIGS=$(curl -s -b "$COOKIE_JAR" "$OPENLIT_URL/api/db-config")
case "$EXISTING_CONFIGS" in
  *'"name":"Default DB"'*)
    echo "Default DB already registered, nothing to do."
    ;;
  *)
    curl -sf -b "$COOKIE_JAR" -X POST "$OPENLIT_URL/api/db-config" \
      -H "Content-Type: application/json" \
      -H "Origin: $OPENLIT_URL" \
      -d "{\"name\":\"Default DB\",\"environment\":\"production\",\"username\":\"$INIT_DB_USERNAME\",\"password\":\"$INIT_DB_PASSWORD\",\"host\":\"$INIT_DB_HOST\",\"port\":\"$INIT_DB_PORT\",\"database\":\"$INIT_DB_DATABASE\"}" \
      >/dev/null
    echo "Registered ClickHouse as OpenLIT's Default DB."
    ;;
esac
