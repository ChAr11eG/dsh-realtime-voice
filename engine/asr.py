"""ASR providers: qwen3 (local Qwen3-ASR) and openai (cloud). All calls run on one worker thread."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
import time

import httpx
import numpy as np
import soundfile as sf

RATE = 16000
LANGUAGES = {"zh": "Chinese", "en": "English"}
graph_installed = False


class Qwen3:
    def __init__(self, cfg):
        self.model_name = cfg.get("model") or "Qwen/Qwen3-ASR-0.6B"
        self.device = cfg.get("device") or "cuda"
        self.model = None

    def load(self):
        global graph_installed
        import torch
        from qwen_asr import Qwen3ASRModel
        import asr_graph

        if not graph_installed:
            asr_graph.install()
            graph_installed = True
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_name, dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
            device_map=self.device, attn_implementation="sdpa", max_new_tokens=128, max_inference_batch_size=1)
        self.transcribe(np.zeros(RATE, np.float32), None)  # captures the CUDA-graph decode step

    def transcribe(self, audio, language):
        return self.model.transcribe(audio=(audio, RATE), language=LANGUAGES.get(language))[0].text.strip()

    def unload(self):
        import torch
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


class OpenAI:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        pass

    def transcribe(self, audio, language):
        cfg = self.cfg
        wav = io.BytesIO()
        sf.write(wav, audio, RATE, format="WAV", subtype="PCM_16")
        data = {"model": cfg.get("model") or "whisper-1"}
        if language in LANGUAGES:
            data["language"] = language
        headers = {"Authorization": f"Bearer {cfg['apiKey']}"} if cfg.get("apiKey") else {}
        response = httpx.post(cfg.get("baseUrl", "").rstrip("/") + "/audio/transcriptions", data=data,
                              files={"file": ("audio.wav", wav.getvalue(), "audio/wav")}, headers=headers, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"云端识别失败（{response.status_code}）：{response.text[:300]}")
        return response.json().get("text", "").strip()

    def unload(self):
        pass


class ASR:
    """The selected provider, loaded on the worker thread by warm() or the first transcription."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(1, "asr")
        self.provider = None
        self.key = None
        self.name = None
        self.language = "auto"
        self.pending = 0

    def configure(self, cfg):
        self.language = cfg.get("language", "auto")
        name = cfg["provider"]
        key = json.dumps([name, cfg.get(name)], sort_keys=True)
        if key == self.key:
            return
        old = self.provider
        self.key, self.name = key, name
        self.provider = {"qwen3": Qwen3, "openai": OpenAI}[name](cfg.get(name) or {})
        self.provider.ready, self.provider.loading, self.provider.error = False, False, None
        if old is not None:
            self.pool.submit(old.unload)

    def health(self):
        provider = self.provider
        return {"provider": self.name, "ready": provider.ready, "loading": provider.loading, "error": provider.error}

    @property
    def busy(self):
        return self.pending > 0

    def submit(self, audio=None):
        """Future of (text, seconds) for PCM16 16 kHz `audio`; None only loads the provider."""
        self.pending += 1
        future = asyncio.get_running_loop().run_in_executor(self.pool, self._work, self.provider, audio)
        future.add_done_callback(lambda _: setattr(self, "pending", self.pending - 1))
        return future

    def _work(self, provider, audio):
        if not provider.ready:
            provider.loading = True
            try:
                provider.load()
                provider.ready, provider.error = True, None
            except Exception as exc:
                provider.error = str(exc) or type(exc).__name__
                raise
            finally:
                provider.loading = False
        if audio is None:
            return None
        started = time.monotonic()
        text = provider.transcribe(audio.astype(np.float32) / 32768, self.language)
        return text, time.monotonic() - started
