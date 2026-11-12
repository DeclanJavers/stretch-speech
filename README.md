# Stretch speech module (STT → LLM → TTS)

Runs the "Talking to new person" state: greets the visitor, chats, gets their name and the item they want, and returns a decision the state machine can act on.

```
mic → silero VAD (detects when they stop talking) → faster-whisper base.en → LLM via llama.cpp or LM Studio (forced JSON) → Kokoro TTS → speakers
```

## Run it

The default model is **LFM3 1.7B**, served by llama.cpp's `llama-server` (`brew install llama.cpp`). LM Studio can't run LFM3 yet on this Mac: its MLX engine (1.14.0) doesn't support the architecture, and it has no llama.cpp engine for macOS.

1. Start the model server and leave it running. 2048 tokens of context is plenty: a full 8-turn chat uses about 900.
   ```bash
   llama-server -m ~/.lmstudio/models/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-QAD-Q4_0.gguf --alias lfm3-1.7b -c 2048 --jinja --port 8081
   ```
   `--jinja` is required: it uses the model's own chat template, which `brain.py` switches to no-thinking mode with `enable_thinking: false`.
2. In a second terminal, from this folder (the first run asks for microphone permission: allow it):
   ```bash
   uv run demo.py
   ```
   Press Enter, then talk. Useful flags:
   - `--auto --loop`: hands-free; a new conversation starts each time one ends (best for the video)
   - `--items pen,plushie`: pretend shirts ran out
   - `--voice am_michael`: other voices include `af_heart` (default), `af_bella`, `bf_emma`, and `am_adam`
   - `--text`: type instead of talk; `--no-tts`: don't speak
   - `--mic 4`: pick another input device (list them with `uv run python -m sounddevice`)

Every conversation is saved to `logs/`.

### Using Qwen or LFM2.5 instead (LM Studio)

```bash
lms load qwen/qwen3.5-4b -c 2048
```

```bash
uv run demo.py --model qwen/qwen3.5-4b --base-url http://localhost:1234/v1
```

For LFM2.5, use `lfm2.5-2.6b-mlx` as the model name in both commands. It needs the template patch described below.

## For the team: the interface

```python
from brain import Brain
from stt import Listener
from tts import Speaker
from conversation import talk_to_person

brain, listener, speaker = Brain(), Listener(), Speaker()   # load once, about 10 s
result = talk_to_person(listener, speaker, brain, available_items=["pen", "shirt", "plushie"])
# result = {"outcome": "item" | "declined" | "unclear" | "no_response",
#           "item": "pen" | "shirt" | "plushie" | None,
#           "name": "Declan" | None, "transcript": [...], "turn_timings": [...]}
```

| outcome | diagram transition |
|---|---|
| `item` | Person asks for something → Logging name and item |
| `declined` | Person doesn't want anything → Logging person in memory |
| `unclear` | Talked for 8 turns without choosing (treat as declined) |
| `no_response` | Silent 3 times in a row (person probably left) |

## Evals

```bash
uv run eval/run_eval.py --runs 3           # 70 labeled utterances, default model (LFM3)
LLM_BASE_URL=http://localhost:1234/v1 uv run eval/run_eval.py --model qwen/qwen3.5-4b  # an LM Studio model
```

Measured on an M3 Pro MacBook, 2026-11-11, same prompt for both:

| Model | Runs on | Item accuracy (70 cases) | Names | LLM reply time | Model size |
|---|---|---|---|---|---|
| Qwen 3.5 4B (MLX 4-bit) | LM Studio | **97.1%** (68/70) | 5/5 | 1.6 s | 3.1 GB |
| LFM3 1.7B (GGUF Q4_0) | llama.cpp | 91.0% avg over 3 runs (87–94%) | 15/15 | **0.8 s** | 1.05 GB |
| LFM2.5 2.6B (MLX 4-bit) | LM Studio | 84.3% (59/70) | 5/5 | 1.15 s | 1.6 GB |

LFM3's misses are mostly requests for two items ("a pen and a plushie"), out-of-stock requests, and self-corrections ("Shirt. Actually, plushie."). In a live test it also treated "Hi, I'm Declan. What do you have?" as an order for a pen.

LFM's misses are mostly declines ("I'm all set" read as still deciding) and requests for several items or the wrong item. In a live test it also said it had no t-shirts when asked for a "tea shirt".

| Stage | Before tuning | Now |
|---|---|---|
| End-of-speech pause | 0.8 s | 0.6 s |
| Whisper | 1.05 s (small.en) | 0.3 s (base.en, same transcripts in tests) |
| LLM | 2.3 s (Qwen, 540-token prompt) | 1.15 s (LFM) / 1.6 s (Qwen), 320-token prompt |
| Kokoro first audio | 0.85 s (whole reply) | ~0.2 s (first sentence) |
| **Total felt delay** | **~5 s** | **~2.3 s (LFM) / ~2.7 s (Qwen)** |

The "Now" total is added up from the stage timings and hasn't been measured with a live mic yet. The `reply_latency_s` the demo prints leaves out the end-of-speech pause.

## Things learned the hard way

- **Thinking off in LM Studio** is `reasoning_effort: "none"`. The `chat_template_kwargs: {enable_thinking: false}` trick is ignored. With thinking on, each reply takes ~10 s.
- **Qwen's template rejects a chat that starts with the assistant** ("No user query found"). Stretch speaks first, so `brain.py` inserts a "(A visitor walks up to you.)" user turn.
- **LFM2.5 always thinks** (its template opens `<think>` with no off switch), and LM Studio then files the forced JSON under `reasoning_content`, leaving the answer empty. The fix is a one-line template edit in `~/.lmstudio/models/LiquidAI/LFM2.5-2.6B-MLX-4bit/chat_template.jinja`: `<think>` becomes `<think></think>`. The original is saved next to it as `chat_template.jinja.orig`. Redo this if the model is re-downloaded.
- **Prompt length is the LLM's delay.** At 540 tokens the first word took 1.4 s; at ~200 it took 0.4 s. Forcing JSON costs nothing. Context length (the `-c` cap) mainly sets how much memory is reserved, not speed.
- **Small models label more reliably when they write the sentence first**, and when the labels are `declined`/`undecided` instead of `"none"`/`null`. Writing the sentence first took LFM from 71% to 80%, and the clearer labels took it to 84%. Renaming the field from `item` to `decision` dropped it to 50%: it answered "pen" to nearly everything.
- `response_format: json_schema` makes the JSON always parse, and `item` can only be a legal value. It costs almost no speed.

## Next steps

- Cut latency further: stream the LLM reply and start speaking after the first sentence of `say`.
- Record real crowd noise and re-run the STT test; raise `VAD_THRESHOLD` if it triggers on background chatter.
- Audio on Stretch: run this on a laptop and either use a Bluetooth speaker/mic on the robot or stream audio over ROS.
