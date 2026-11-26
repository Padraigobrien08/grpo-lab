#!/usr/bin/env bash
# Snapshot the local run store, build the static dashboard and deploy it to Vercel.
#
#   bash scripts/deploy_dashboard.sh            # production
#   bash scripts/deploy_dashboard.sh --preview  # preview URL only
#
# Deploys the prebuilt web/out/ directly: the snapshot (web/public/data/) is built from the local
# runs/ store, which isn't in git, so a git-connected Vercel build couldn't produce it.
# First time only: `npx vercel@65.14.0 login` (the deploy then asks to create/link the project).
set -euo pipefail
cd "$(dirname "$0")/.."

VERCEL="npx --yes vercel@65.14.0"
TARGET="--prod"
[[ "${1:-}" == "--preview" ]] && TARGET=""

uv run export-static
(cd web && pnpm build)

# Next's export writes run.html / rollouts.html; cleanUrls serves them at /run and /rollouts,
# which is what the app links to.
echo '{"cleanUrls": true}' > web/out/vercel.json

$VERCEL deploy web/out $TARGET
