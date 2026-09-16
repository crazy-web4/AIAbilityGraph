# ────────────────────────────────────────────────────────────────────
# Edge TTS provider — uses Microsoft Edge's free TTS via edge-tts.
#
# Docs:    https://github.com/rany2/edge-tts
# Install: pip install edge-tts (already installed)
# Voices:  zh-CN-YunxiNeural (warm male, default)
#          zh-CN-YunjianNeural (narration)
#          zh-CN-YunhaoNeural (calm male)
#          zh-CN-XiaoxiaoNeural (female)
#
# Strengths: Free, no API key, high quality Chinese voices
# ────────────────────────────────────────────────────────────────────

tts_check() {
  if ! command -v edge-tts >/dev/null; then
    echo "✗ edge-tts not found." >&2
    echo "  Install: pip3 install edge-tts" >&2
    return 1
  fi
}

tts_install_help() {
  cat <<'EOF' >&2
To use Edge TTS:

  Install:  pip3 install edge-tts

Available Chinese voices:
  • zh-CN-YunxiNeural   — 温暖男声 (warm male, default)
  • zh-CN-YunjianNeural — 解说男声 (narration)
  • zh-CN-YunhaoNeural  — 沉稳男声 (calm male)
  • zh-CN-XiaoxiaoNeural — 阳光女声 (female)

Examples:
  PRESENTATION_TTS=edge-tts npm run synthesize-audio
  PRESENTATION_TTS=edge-tts PRESENTATION_TTS_VOICE=zh-CN-YunjianNeural npm run synthesize-audio
EOF
}

tts_synthesize() {
  local text="$1"
  local out="$2"
  local voice="${3:-zh-CN-YunxiNeural}"

  # Default voice: zh-CN-YunxiNeural (warm male Chinese voice)
  edge-tts --voice "$voice" --text "$text" --write-media "$out" \
    >/dev/null 2>&1
}
