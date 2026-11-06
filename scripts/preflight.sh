#!/usr/bin/env bash
# Repository gate. Runs from any directory and never modifies the worktree.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

failed=()
skipped=()

step() { printf '\n=== %s ===\n' "$1"; }
pass() { printf '[ OK ] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; failed+=("$1"); }
skip() { printf '[SKIP] %s: %s\n' "$1" "$2"; skipped+=("$1"); }

step "compileall"
if python3 -m compileall -q deploy src scripts tests; then pass "compileall"; else fail "compileall"; fi

if [ -f "$ROOT/.venv/bin/python" ]; then
  for target in check-code typecheck pytest; do
    step "make $target"
    if make --no-print-directory "$target"; then pass "make $target"; else fail "make $target"; fi
  done
else
  for target in check-code typecheck pytest; do
    skip "make $target" "no .venv; run make setup"
  done
fi

step "complexity hotspots (report only)"
if [ -x "$ROOT/.venv/bin/ruff" ]; then
  "$ROOT/.venv/bin/ruff" check --select PLR0915,C901 deploy src --output-format concise 2>&1 | sed 's/^/  /' || true
  pass "complexity hotspots reported above"
else
  skip "complexity hotspots" "no .venv; run make setup"
fi

step "make version-check"
if make --no-print-directory version-check; then pass "make version-check"; else fail "make version-check"; fi

step "git diff --check"
if git diff HEAD --check; then pass "git diff --check"; else fail "git diff --check"; fi

printf '\n=== summary ===\n'
if [ ${#skipped[@]} -gt 0 ]; then printf 'skipped: %s\n' "${skipped[*]}"; fi
if [ ${#failed[@]} -gt 0 ]; then
  printf 'failed: %s\n' "${failed[*]}"
  exit 1
fi
printf 'all executed checks passed\n'
