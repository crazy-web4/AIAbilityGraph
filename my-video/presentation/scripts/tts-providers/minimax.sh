# ────────────────────────────────────────────────────────────────────
# MiniMax TTS provider — uses the official mmx-cli.
#
# Docs:  https://platform.minimaxi.com/docs/token-plan/minimax-cli
# Repo:  https://github.com/MiniMax-AI/cli
#
# Strengths: Chinese narration quality is consistently good; lots of
# voice options; one-line CLI call.
# Voices:  male-alt (warm male, default) / male-home / male-shaon / female-***
# ────────────────────────────────────────────────────────────────────

tts_check() {
  if ! command -v mmx >/dev/null; then
    echo "✗ mmx CLI not found in PATH." >&2
    echo "  Install: npm install -g mmx-cli" >&2
    return 1
  fi
  if ! mmx auth status >/dev/null 2>&1; then
    echo "✗ mmx is not authenticated." >&2
    echo "  Login: mmx auth login --api-key sk-xxxxx" >&2
    echo "  Get API key: https://platform.minimaxi.com" >&2
    return 1
  fi
}

tts_install_help() {
  cat <<'EOF' >&2
To use the MiniMax provider:

  Install:  npm install -g mmx-cli
  Login:    mmx auth login --api-key sk-xxxxx
            (get a key at https://platform.minimaxi.com)

Recommended voices for warm male Chinese narration:
  • male-alt      — 温暖男声，适合叙事
  • male-home     — 沉稳男声
  • male-shaon    — 青年男声

Or pick another provider: PRESENTATION_TTS=<name> npm run synthesize-audio
See tts-providers/README.md for the list and how to add your own.
EOF
}

tts_synthesize() {
  local text="$1"
  local out="$2"
  local voice="${3:-male-qn-qingse}"

  # Default voice: male-qn-qingse (warm, gentle male voice for Chinese narration)
  # Other options: male-qn-jingying (elite), male-qn-badao (dominant), male-qn-daxuesheng (college student)
  # Add -jingpin suffix for premium quality (costs more credits)
  mmx speech synthesize \
    --voice "$voice" \
    --text "$text" \
    --out "$out" \
    >/dev/null 2>&1
}
