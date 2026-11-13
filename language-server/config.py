"""All the knobs for the speech module in one place."""

import os

# --- Items Stretch can hand out ------------------------------------------------
# The state machine can pass a smaller list to talk_to_person() when one runs out.
ALL_ITEMS = ["pen", "shirt", "plushie"]

# How each item is described to people (and to the LLM).
ITEM_DESCRIPTIONS = {
    "pen": "a pen",
    "shirt": "a t-shirt",
    "plushie": "a plushie (a little stuffed animal)",
}

# --- LLM (LM Studio's OpenAI-compatible server) --------------------------------
# Default is llama.cpp's llama-server on :8081 (LFM3 can't run in LM Studio yet).
# LM Studio is http://localhost:1234/v1.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8081/v1")
# lfm3-1.7b: 91% on the eval, 0.8 s per reply (llama.cpp).
# qwen/qwen3.5-4b: 97%, 1.6 s.  lfm2.5-2.6b-mlx: 84%, 1.15 s (both LM Studio).
LLM_MODEL = os.environ.get("LLM_MODEL", "lfm3-1.7b")
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 200
LLM_TIMEOUT_S = 20

# --- Speech-to-text (faster-whisper) -------------------------------------------
# base.en: 0.3 s per utterance vs 1.0 s for small.en, same transcripts in our tests.
# Switch back to small.en if a noisy room causes mistakes.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = "cpu"          # Apple Silicon: CPU + int8 is the fast path for CTranslate2
WHISPER_COMPUTE = "int8"
# Nudges Whisper toward the words we care about (helps a lot with "plushie").
WHISPER_PROMPT = "Stretch, pen, t-shirt, shirt, plushie, stuffed animal, no thanks."

# --- Listening / turn-taking (silero VAD) --------------------------------------
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5             # raise toward 0.7 in a loud room
END_OF_SPEECH_SILENCE_MS = 600  # how long a pause means "they're done talking"
WAIT_FOR_SPEECH_S = 10          # give up if nobody says anything for this long
MAX_UTTERANCE_S = 15            # cut off very long rambles

# --- Text-to-speech (Kokoro) ---------------------------------------------------
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
KOKORO_SPEED = 1.05
TTS_SAMPLE_RATE = 24000

# --- Conversation --------------------------------------------------------------
MAX_TURNS = 8                   # open chat, but don't let one person hold the robot forever
MAX_SILENT_TURNS = 2            # "are you still there?" this many times, then give up
