# Shared env for HTTP smoke scripts. Source from scripts/smoke*.sh — do not run directly.
#
# Requires repo-root .env (copy from .env.example):
#   MODAL_WEB_URL, MODAL_STREAM_URL — lunas app (label APP_NAME / APP_NAME-stream)
#   MODAL_KEY, MODAL_SECRET — proxy auth from modal.com/settings/proxy-auth-tokens
#
# CLI smokes without HTTP auth: uv run modal run infusers/modal_app/...::smoke

_modal_env_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

_modal_env_load() {
  REPO_ROOT="$(_modal_env_repo_root)"
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi
  MODAL_KEY="${MODAL_KEY:?Set MODAL_KEY in .env (wk- proxy token ID)}"
  MODAL_SECRET="${MODAL_SECRET:?Set MODAL_SECRET in .env (ws- proxy token secret)}"
  MODAL_AUTH_HEADERS=(
    -H "Modal-Key: ${MODAL_KEY}"
    -H "Modal-Secret: ${MODAL_SECRET}"
  )
}
