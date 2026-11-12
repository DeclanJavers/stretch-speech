"""Listening: microphone -> voice activity detection -> Whisper transcript.

listen() blocks until the person starts talking, records until they pause
for END_OF_SPEECH_SILENCE_MS, then transcribes. Returns None if nobody spoke.
"""

import collections
import queue
import time
import warnings

import numpy as np
import sounddevice as sd

import config

warnings.filterwarnings("ignore")

CHUNK = 512  # silero VAD wants 512-sample chunks at 16 kHz (32 ms)


class Listener:
    def __init__(self, input_device=None):
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad

        self.input_device = input_device
        self.whisper = WhisperModel(
            config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE
        )
        self.vad_model = load_silero_vad()

    # ---- recording ---------------------------------------------------------
    def record_utterance(self) -> np.ndarray | None:
        """Wait for speech, return the audio of one utterance (float32, 16 kHz)."""
        from silero_vad import VADIterator

        vad = VADIterator(
            self.vad_model,
            threshold=config.VAD_THRESHOLD,
            sampling_rate=config.SAMPLE_RATE,
            min_silence_duration_ms=config.END_OF_SPEECH_SILENCE_MS,
            speech_pad_ms=100,
        )
        audio_q: queue.Queue = queue.Queue()

        def callback(indata, frames, t, status):
            audio_q.put(indata[:, 0].copy())

        preroll = collections.deque(maxlen=int(0.5 * config.SAMPLE_RATE / CHUNK))  # keep 0.5 s before speech
        recorded: list[np.ndarray] = []
        speaking = False
        start = time.monotonic()
        speech_start = None

        with sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=CHUNK, device=self.input_device, callback=callback,
        ):
            while True:
                chunk = audio_q.get()
                if len(chunk) != CHUNK:
                    continue
                event = vad(chunk, return_seconds=False)

                if speaking:
                    recorded.append(chunk)
                    too_long = time.monotonic() - speech_start > config.MAX_UTTERANCE_S
                    if (event and "end" in event) or too_long:
                        break
                else:
                    preroll.append(chunk)
                    if event and "start" in event:
                        speaking = True
                        speech_start = time.monotonic()
                        recorded.extend(preroll)
                    elif time.monotonic() - start > config.WAIT_FOR_SPEECH_S:
                        return None

        vad.reset_states()
        return np.concatenate(recorded)

    # ---- transcription -----------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.whisper.transcribe(
            audio,
            language="en",
            beam_size=5,
            initial_prompt=config.WHISPER_PROMPT,
            condition_on_previous_text=False,
            vad_filter=False,  # we already cut the audio with silero
        )
        # Whisper "hears" phantom words ("Thank you.", "you") in noise. Drop segments it
        # is itself unsure contain speech.
        kept = [s.text.strip() for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
        return " ".join(kept).strip()

    def listen(self) -> tuple[str | None, dict]:
        """Record one utterance and transcribe it. Returns (text or None, timings)."""
        audio = self.record_utterance()
        if audio is None:
            return None, {}
        t = time.perf_counter()
        text = self.transcribe(audio)
        timings = {"stt_s": round(time.perf_counter() - t, 2), "utterance_s": round(len(audio) / config.SAMPLE_RATE, 1)}
        return (text or None), timings
