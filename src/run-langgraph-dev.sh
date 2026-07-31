#!/usr/bin/env bash

set -euo pipefail

# NOTE: model credentials (LITELLM_KEY, etc.) are read from ../.env via
# langgraph.json's "env" field, not from the shell -- langgraph dev loads
# that file itself and it takes precedence over shell-exported values, so
# there is no point exporting secrets here.

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_file="$project_dir/langgraph.json"

if [[ -x "$project_dir/../.venv/bin/langgraph" ]]; then
  langgraph_bin="$project_dir/../.venv/bin/langgraph"
elif command -v langgraph >/dev/null 2>&1; then
  langgraph_bin="$(command -v langgraph)"
else
  printf 'Error: langgraph was not found in .venv or PATH.\n' >&2
  exit 1
fi

cd "$project_dir"
exec "$langgraph_bin" dev --config "$config_file" "$@"
