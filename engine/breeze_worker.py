"""Breeze TTS 2 worker for the speech engine. Runs in Breeze's own venv (torch 2.9.1, transformers 4.57.3);
with the voice reference given per request."""

import argparse
import asyncio
from contextlib import asynccontextmanager, closing
from dataclasses import replace
import io
import logging
import os
from pathlib import Path
from queue import Queue
import sys
from threading import Event, Lock, Thread
import time
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("breeze")
SEED = 42
# Spoken replies are a few seconds; 500 codec frames is 40 s of audio. Smaller caches keep GPU memory down.
# The prompt (up to a 20 s reference plus its transcript and a 200-character phrase) fits the 640 prefill
# bucket in breeze-fast.json; 640 + 500 fits the cache.
MAX_NEW_TOKENS = 500
MAX_SEQ_LEN = 1280
args = None


@asynccontextmanager
async def lifespan(app):
    sys.path.insert(0, args.code_dir)
    from breeze_infer import templates
    from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
    from models.warmup_profile import load_warmup_profile

    tokenizer, model, audio_tokenizer = load_runtime(Path(args.model), device=resolve_device(),
                                                     attn_implementation="eager")
    update_generation_config_for_breeze(model)
    # Encode each reference once (keyed by path and mtime) instead of on every request.
    encode, encoded = templates._encode_prompt_audio, {}

    def cached_encode(audio_tokenizer, path):
        key = (path, os.stat(path).st_mtime_ns)
        if key not in encoded:
            encoded[key] = encode(audio_tokenizer, path)
        return encoded[key]
    templates._encode_prompt_audio = cached_encode
    app.state.tokenizer, app.state.model, app.state.audio_tokenizer = tokenizer, model, audio_tokenizer
    app.state.runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer,
        FastStreamingConfig(max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN, fast_all=True),
        tokenizer=tokenizer)
    app.state.design_runtime = None
    # CUDA-graph fast path only: eager generation measured slower than real time. Clone-only
    # warmup profile (no CFG branch, backbone batch 1, prefill up to 640) adapted from breeze-tts/configs/fast.json.
    profile = replace(load_warmup_profile(ROOT / "breeze-fast.json"),
                      codec_chunk_frames=app.state.runtime.codec_chunk_frames)
    manifest = app.state.runtime.warmup_from_profile(profile)
    logger.info("fast warmup: %.0f ms", manifest["total_elapsed_ms"])
    app.state.lock = Lock()
    for text in ("Warm up.", "热身。"):  # lazy cache allocation
        for _ in generate({"id": "voice", "text": text, "speaker": "S0"}):
            pass
    yield


app = FastAPI(lifespan=lifespan)


class CancellableStream(StreamingResponse):
    """Calls cancel however the response ends, including a client disconnect during send()."""

    def __init__(self, content, cancel, **kwargs):
        super().__init__(content, **kwargs)
        self.cancel = cancel

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.cancel()


class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    ref_audio: str
    ref_text: str
    language: Literal["en", "zh"] | None = None


class Design(BaseModel):
    instruction: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000)
    seed: int = SEED
    cfg: float = 4.0


def generate(request, cancelled=None):
    """Yield PCM16 chunks as the model produces them; the lock serialises requests."""
    from breeze_infer.runtime import set_all_seeds
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name

    with app.state.lock:
        started = time.monotonic()
        set_all_seeds(SEED)
        inputs = prepare_inputs(app.state.tokenizer, app.state.audio_tokenizer, app.state.model, [request],
                                get_template(select_template_name(request)),
                                guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None)
        prepared = time.monotonic() - started
        first, samples = None, 0
        # Closing the iterator releases the runtime's per-request codec state.
        with closing(app.state.runtime.iter_audio_chunks(inputs, request_id="voice", seed=SEED)) as stream:
            for chunk in stream:
                if cancelled is not None and cancelled.is_set():
                    break
                audio = np.asarray(chunk.audio, dtype=np.float32)
                if not audio.size or not np.isfinite(audio).all():
                    continue
                if first is None:
                    first = time.monotonic() - started
                samples += audio.size
                yield (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
            generated = time.monotonic() - started
        logger.info("%d chars: prepared %.3fs, first chunk %.3fs, %.2fs audio in %.3fs, closed after %.3fs",
                    len(request["text"]), prepared, first or -1, samples / app.state.runtime.sample_rate,
                    generated, time.monotonic() - started)


@app.get("/health")
def health():
    return {"status": "ok", "model": "BreezeBlue/Breeze-TTS-2", "sample_rate": 24000}


@app.post("/synthesize")
async def synthesize(request: Speech):
    clone = {"id": "voice", "text": request.text, "speaker": "S0", "ref_audio_path": request.ref_audio,
             "ref_text": request.ref_text}
    # Generation runs in its own thread, never suspended inside the model lock: a client that
    # disconnects mid-stream (an interrupted reply) cancels it instead of holding the lock forever.
    chunks, cancelled = Queue(), Event()

    def produce():
        try:
            for pcm in generate(clone, cancelled):
                chunks.put(pcm)
        except Exception as exc:
            logger.exception("Synthesis failed")
            chunks.put(exc)
        finally:
            chunks.put(None)

    Thread(target=produce, daemon=True).start()
    first = await asyncio.to_thread(chunks.get)
    if isinstance(first, Exception) or first is None:  # failed before any audio: a plain error reply
        return JSONResponse({"detail": str(first or "no audio")}, 500)

    async def stream():
        yield first
        while (pcm := await asyncio.to_thread(chunks.get)) is not None:
            if isinstance(pcm, Exception):
                raise pcm
            yield pcm

    return CancellableStream(stream(), cancelled.set, media_type="application/octet-stream",
                             headers={"X-Sample-Rate": "24000"})


@app.post("/design")
def design(request: Design):
    """Voice design (instruction only) on a separate eager runtime: the fast one is frozen at CFG 1."""
    import soundfile as sf
    from breeze_infer.runtime import set_all_seeds
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    model = app.state.model
    with app.state.lock:
        try:
            if app.state.design_runtime is None:
                app.state.design_runtime = FastBreezeStreamingRuntime(
                    model, app.state.audio_tokenizer, FastStreamingConfig(max_new_tokens=1500, max_seq_len=2048),
                    tokenizer=app.state.tokenizer)
            # The text-encoder CUDA-graph switch lives on the model; the warmed graphs do not cover CFG 4.
            model._fast_text_encoder_cudagraph = False
            started = time.monotonic()
            spec = {"id": "design", "text": request.text, "speaker": "S0", "instruction": request.instruction}
            set_all_seeds(request.seed)
            inputs = prepare_inputs(app.state.tokenizer, app.state.audio_tokenizer, model, [spec],
                                    get_template(select_template_name(spec)),
                                    guidance_scale=request.cfg, guidance_scale_ref=None, guidance_scale_ins=None)
            with closing(app.state.design_runtime.iter_audio_chunks(inputs, request_id="design",
                                                                    seed=request.seed)) as stream:
                audio = np.concatenate([np.asarray(chunk.audio, dtype=np.float32) for chunk in stream])
        finally:
            model._fast_text_encoder_cudagraph = True
    logger.info("design: %.2fs audio in %.2fs", audio.size / 24000, time.monotonic() - started)
    wav = io.BytesIO()
    sf.write(wav, np.clip(np.nan_to_num(audio), -1, 1), app.state.design_runtime.sample_rate, format="WAV",
             subtype="PCM_16")
    return Response(wav.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=47911)
    parser.add_argument("--model", required=True, help="Breeze-TTS-2 model directory")
    parser.add_argument("--code-dir", required=True, help="breeze-tts checkout (breeze_infer, models)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
