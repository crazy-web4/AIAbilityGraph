# ────────────────────────────────────────────────────────────────────
# macOS say provider — uses built-in `say` command.
#
# Docs:    man say
# Env:     No API key required — built into macOS
# Voices:  Use --voice to pick voice (default: Ting-Ting for Chinese)
#
# Strengths: free, offline, no setup required
# Weaknesses: robotic sound, limited expressiveness
# ────────────────────────────────────────────────────────────────────

tts_check() {
  if ! command -v say >/dev/null; then
    echo "✗ macOS say command not found." >&2
    return 1
  fi
  if ! command -v ffmpeg >/dev/null; then
    echo "✗ ffmpeg not found (required for MP3 conversion)." >&2
    echo "  Install: brew install ffmpeg" >&2
    return 1
  fi
}

tts_install_help() {
  cat <<'EOF' >&2
macOS say is built-in — no installation needed.

Install ffmpeg for MP3 conversion:
  brew install ffmpeg

For Chinese narration, recommended voices:
  • Ting-Ting (Mandarin, female)
  • Mei-Jia (Mandarin, female)
  • Sin-Ji (Cantonese)

Or pick another provider: PRESENTATION_TTS=<name> npm run synthesize-audio
EOF
}

tts_synthesize() {
  local text="$1"
  local out="$2"
  local voice="${3:-Ting-Ting}"

  # Create a temp AIFF file
  local temp_aiff="${out%.mp3}.temp.aiff"

  # Generate AIFF with say command
  if ! say -v "$voice" -o "$temp_aiff" "$text" 2>/dev/null; then
    rm -f "$temp_aiff" 2>/dev/null
    return 1
  fi

  # Convert AIFF to MP3 using ffmpeg
  if ffmpeg -y -i "$temp_aiff" -acodec libmp3lame -qscale:a 2 "$out" >/dev/null 2>&1; then
    rm -f "$temp_aiff"
    return 0
  else
    # Fallback: if ffmpeg fails, keep AIFF but rename to .mp3
    # Most browsers can still play AIFF
    mv "$temp_aiff" "$out"
    return 0
  fi
}
