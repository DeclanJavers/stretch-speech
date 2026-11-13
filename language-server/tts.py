"""Text-to-speech with Kokoro.

Kokoro splits text into sentences and yields audio for each one. We play
sentence 1 while sentence 2 is still being synthesized, so Stretch starts
talking sooner on long replies.
"""

import queue
import threading
import time
import warnings

import numpy as np
import sounddevice as sd

import config

warnings.filterwarnings("ignore")


class Speaker:
    def __init__(self, voice: str = config.KOKORO_VOICE, enabled: bool = True):
        self.voice = voice
        self.enabled = enabled
        self.pipeline = None
        if enabled:
            from kokoro import KPipeline  # slow import, only when actually speaking

            self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
            # Warm up so the first real sentence isn't slow.
            for _ in self.pipeline("Hi.", voice=self.voice):
                pass

    def say(self, text: str) -> dict:
        """Speak text out loud and block until finished. Returns timing info."""
        if not self.enabled or not text.strip():
            return {"tts_first_audio_s": 0.0, "tts_total_s": 0.0}

        chunks: queue.Queue = queue.Queue()
        start = time.perf_counter()

        def produce():
            # Split per sentence: by default Kokoro renders the whole reply before any audio
            # (0.66 s); per sentence the first audio is ready in ~0.2 s.
            for _, _, audio in self.pipeline(text, voice=self.voice, speed=config.KOKORO_SPEED,
                                             split_pattern=r"(?<=[.!?])\s+"):
                if audio is not None:
                    chunks.put(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
            chunks.put(None)

        threading.Thread(target=produce, daemon=True).start()

        first_audio = None
        while (chunk := chunks.get()) is not None:
            if first_audio is None:
                first_audio = time.perf_counter() - start
            sd.play(chunk.astype(np.float32), config.TTS_SAMPLE_RATE)
            sd.wait()

        return {
            "tts_first_audio_s": round(first_audio or 0.0, 2),
            "tts_total_s": round(time.perf_counter() - start, 2),
        }
