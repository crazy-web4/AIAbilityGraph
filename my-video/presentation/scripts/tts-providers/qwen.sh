# ────────────────────────────────────────────────────────────────────
# Qwen3-TTS provider — uses moma.hq.cmcc Qwen3-TTS service.
#
# Model: qwen/qwen3-tts
# API:   http://moma.hq.cmcc/largemodel/moma/api/v1/audio/speech
# API Key: JWT token (set via QWEN_TTS_API_KEY env or Authorization header)
#
# Strengths: High quality Chinese TTS, supports multiple languages/dialects
# Voices:  Built-in 49 native voices (model handles auto-selection)
# ────────────────────────────────────────────────────────────────────

tts_check() {
  if ! command -v curl >/dev/null; then
    echo "✗ curl not found." >&2
    return 1
  fi
  if ! command -v jq >/dev/null; then
    echo "✗ jq not found." >&2
    return 1
  fi
  # API key check
  if [[ -z "${QWEN_TTS_API_KEY:-}" ]]; then
    echo "⚠ QWEN_TTS_API_KEY not set, using default JWT token." >&2
  fi
  return 0
}

tts_install_help() {
  cat <<'EOF' >&2
Qwen3-TTS provider configured for:
  Base URL: http://moma.hq.cmcc/largemodel/moma/api/v1/audio/speech
  Model: qwen/qwen3-tts

Required:
  • API Key (JWT token) — set via QWEN_TTS_API_KEY env var

Optional:
  • QWEN_TTS_VOICE — Voice selection (model auto-selects if not specified)

Run synthesis:
  PRESENTATION_TTS=qwen PRESENTATION_TTS_VOICE=male npm run synthesize-audio
EOF
}

tts_synthesize() {
  local text="$1"
  local out="$2"
  local voice="${3:-}"

  local base_url="http://moma.hq.cmcc/largemodel/moma/api/v1/audio/speech"
  local api_key="${QWEN_TTS_API_KEY:-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhcGlfa2V5IjoiNjgwNzE4YzcxZDdhMTQzNmY3ZTRkOWQ3IiwiZXhwIjoxODU1NDQzNTk1MTQzMzIsInRpbWVzdGFtcCI6MTc3MjQxMzUzMn0.CKONlwCIWpu1iefavi8yMniOdBsKx5rxWt7R39jfO0o}"

  # Build JSON payload
  local payload
  payload=$(jq -n \
    --arg model "qwen/qwen3-tts" \
    --arg input "$text" \
    --arg format "mp3" \
    '{model: $model, input: $input, response_format: $format}')

  # Call API with accept header for audio
  curl -s -o "$out" -X POST "$base_url" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $api_key" \
    -H "Accept: audio/mpeg" \
    -d "$payload" 2>/dev/null

  # Check if output is valid audio (non-empty and not JSON error)
  if [[ -s "$out" ]] && ! file "$out" | grep -q JSON; then
    return 0
  else
    rm -f "$out" 2>/dev/null
    return 1
  fi
}
