"""Speech engine for dsh-realtime-voice: turn-taking (Silero VAD), ASR, TTS and the voice store."""

import os

# Lets torch return the ASR's GPU memory when the provider is switched away (before torch is imported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import asyncio
import copy
from contextlib import asynccontextmanager
import io
import json
import logging
from pathlib import Path
import random
import tempfile
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf
import uvicorn

from asr import ASR
from tts import TTS, TTSError, language_of
from vad import Segmenter
from voices import Voices, decode

logger = logging.getLogger("engine")
MAX_UPLOAD = 30 * 1024 * 1024
# Spoken by a designed voice and kept as its clone transcript when the request has no text.
DESIGN_TEXT = {
    "zh": "你好，很高兴认识你。今天天气不错，我们随便聊聊吧，有什么想说的都可以告诉我。",
    "en": "Hello there, it's nice to meet you. The weather is lovely today, so tell me what's on your mind.",
}
DEFAULT_CONFIG = {
    "asr": {"provider": "qwen3", "language": "auto", "partials": True, "partialInterval": 0.25,
            "qwen3": {"model": "Qwen/Qwen3-ASR-0.6B", "device": "cuda"},
            "openai": {"baseUrl": "https://api.openai.com/v1", "apiKey": "", "model": "whisper-1"}},
    # End-of-turn timing in seconds; a question gets the shorter wait.
    "turn": {"vadThreshold": 0.5, "silence": 0.4, "confirm": 0.16, "questionSilence": 0.2, "questionConfirm": 0.05,
             "minSpeech": 0.35, "preRoll": 0.6, "earlyTranscribe": 0.2},
    "tts": {"provider": "edge",
            "breeze": {"python": "", "codeDir": "", "model": "", "voiceId": "", "port": 47911},
            "edge": {"voice": "zh-CN-YunxiNeural", "rate": "+0%", "pitch": "+0Hz"},
            "openai": {"baseUrl": "https://api.openai.com/v1", "apiKey": "", "model": "gpt-4o-mini-tts",
                       "voice": "alloy"}},
}
TOKEN = ""


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


@asynccontextmanager
async def lifespan(app):
    app.state.config = copy.deepcopy(DEFAULT_CONFIG)
    app.state.voices = Voices(app.state.data_dir)
    app.state.asr = ASR()
    app.state.tts = TTS(app.state.data_dir, app.state.voices)
    app.state.listening = False
    apply_config()
    yield
    await app.state.tts.close()
    app.state.asr.pool.submit(app.state.asr.provider.unload).result()
    app.state.asr.pool.shutdown()


app = FastAPI(lifespan=lifespan)


class Auth:
    """Every HTTP request carries the x-rv-token header (the WebSocket checks it itself)."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and dict(scope["headers"]).get(b"x-rv-token", b"").decode() != TOKEN:
            return await JSONResponse({"detail": "未授权"}, 401)(scope, receive, send)
        await self.inner(scope, receive, send)


app.add_middleware(Auth)


@app.exception_handler(RequestValidationError)
async def invalid(request, exc):
    detail = "; ".join(f"{'.'.join(map(str, error['loc'][1:]))}: {error['msg']}" for error in exc.errors())
    return JSONResponse({"detail": f"请求无效：{detail}"}, 400)


@app.exception_handler(Exception)
async def failed(request, exc):
    logger.exception("Request failed")
    return JSONResponse({"detail": str(exc) or type(exc).__name__}, 500)


def apply_config():
    config = app.state.config
    app.state.asr.configure(config["asr"])
    app.state.tts.configure(config["tts"])


def health():
    return {"status": "ok", "asr": app.state.asr.health(), "tts": app.state.tts.health(),
            "breeze": app.state.tts.breeze.health()}


@app.get("/health")
async def get_health():
    return health()


@app.post("/config")
async def post_config(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "配置必须是 JSON 对象")
    config = merge(copy.deepcopy(DEFAULT_CONFIG), body)
    if config["asr"]["provider"] not in ("qwen3", "openai") or config["tts"]["provider"] not in ("breeze", "edge", "openai"):
        raise HTTPException(400, "未知的识别或合成服务")
    app.state.config = config
    apply_config()
    return health()


@app.post("/warm")
async def warm():
    results = await asyncio.gather(app.state.asr.submit(), app.state.tts.warm(), return_exceptions=True)
    errors = [str(result) or type(result).__name__ for result in results if isinstance(result, BaseException)]
    if errors:
        return JSONResponse({"detail": "；".join(errors), **health()}, 500)
    return health()


# ---------------------------------------------------------------- ASR stream

class Listener:
    """One /asr connection: VAD turn segmentation, partials, early and final transcription."""

    def __init__(self, ws):
        self.ws = ws
        self.config = app.state.config
        self.asr = app.state.asr
        self.segmenter = Segmenter(self.config["turn"])
        self.lock = asyncio.Lock()
        self.closed = False
        self.utterance = 0  # bumps on each onset and reset: drops stale partials
        self.epoch = 0  # bumps on reset: drops finals in flight
        self.early = None  # final ASR started during the end-of-turn silence
        self.next_partial = 0.0
        self.tasks = set()

    async def send(self, message):
        async with self.lock:
            if not self.closed:
                try:
                    await self.ws.send_json(message)
                except Exception:
                    self.closed = True

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def run(self):
        while True:
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            pcm = message.get("bytes")
            if pcm is not None:
                if not 2 <= len(pcm) <= 6400 or len(pcm) % 2:
                    await self.send({"type": "error", "detail": "音频帧必须是 2–6400 字节的 16 kHz PCM16"})
                    continue
                for event in self.segmenter.feed(pcm):
                    await self.handle(event)
                self.maybe_partial()
            elif message.get("text"):
                try:
                    kind = json.loads(message["text"]).get("type")
                except (ValueError, AttributeError):
                    kind = None
                if kind == "reset":
                    self.segmenter.reset()
                    self.utterance += 1
                    self.epoch += 1
                    self.early = None
        self.closed = True

    async def handle(self, event):
        kind = event[0]
        if kind == "start":
            self.utterance += 1
            self.early = None
            self.next_partial = 0.3
            await self.send({"type": "speech_start"})
        elif kind == "early":
            self.early = self.asr.submit(event[1])
        elif kind == "resume":
            self.early = None
        elif kind == "end":
            job, self.early = self.early, None
            if event[2] < self.config["turn"]["minSpeech"]:
                await self.send({"type": "no_speech"})
            else:
                self.spawn(self.final(job or self.asr.submit(event[1]), self.epoch))

    def maybe_partial(self):
        asr, segmenter = self.config["asr"], self.segmenter
        if (not asr["partials"] or not segmenter.recording or segmenter.elapsed < self.next_partial
                or segmenter.silence_seconds >= self.config["turn"]["earlyTranscribe"] or self.asr.busy):
            return
        self.next_partial = segmenter.elapsed + asr["partialInterval"]
        self.spawn(self.partial(self.asr.submit(segmenter.audio()), self.utterance))

    async def partial(self, job, utterance):
        try:
            text, _ = await job
        except Exception as exc:
            await self.send({"type": "error", "detail": f"语音识别失败：{exc}"})
            return
        if text and utterance == self.utterance and self.segmenter.recording:
            self.segmenter.question = text.rstrip().endswith(("?", "？"))
            await self.send({"type": "partial", "text": text})

    async def final(self, job, epoch):
        try:
            text, seconds = await job
        except Exception as exc:
            await self.send({"type": "error", "detail": f"语音识别失败：{exc}"})
            return
        if epoch != self.epoch:
            return
        if text:
            await self.send({"type": "final", "text": text, "asr_seconds": round(seconds, 3)})
        else:  # a cough or noise must not leave the host waiting for words that never come
            await self.send({"type": "no_speech"})


@app.websocket("/asr")
async def asr_socket(ws: WebSocket):
    await ws.accept()
    if ws.headers.get("x-rv-token") != TOKEN:
        await ws.close(4401)
        return
    if app.state.listening:
        await ws.close(4409)
        return
    app.state.listening = True
    listener = None
    try:
        listener = Listener(ws)
        await listener.run()
    except Exception as exc:
        logger.exception("ASR stream failed")
        if listener is not None:
            await listener.send({"type": "error", "detail": str(exc)})
    finally:
        app.state.listening = False
        if listener is not None:
            listener.closed = True
            for task in list(listener.tasks):
                task.cancel()


# ---------------------------------------------------------------- TTS

class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: Literal["zh", "en"] | None = None
    provider: Literal["breeze", "edge", "openai"] | None = None
    voice: str | None = None


class PcmStream(StreamingResponse):
    """Closes the synthesis stream however the response ends, so a disconnect cancels it."""

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.body_iterator.aclose()


@app.post("/tts")
async def tts(speech: Speech):
    started = time.monotonic()
    try:
        stream = app.state.tts.stream(speech.text, speech.language, speech.provider, speech.voice)
        first = await anext(stream)
    except StopAsyncIteration:
        raise HTTPException(502, "语音合成没有产生音频")
    except KeyError as exc:
        raise HTTPException(404, exc.args[0])
    except TTSError as exc:
        raise HTTPException(502, str(exc))
    logger.info("tts %s: first chunk %.3fs", speech.provider or app.state.config["tts"]["provider"],
                time.monotonic() - started)

    async def body():
        try:
            yield first
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    return PcmStream(body(), media_type="application/octet-stream", headers={"X-Sample-Rate": "24000"})


@app.get("/tts/voices")
async def tts_voices(provider: str = "edge"):
    return {"voices": await app.state.tts.edge_catalog() if provider == "edge" else []}


# ---------------------------------------------------------------- voice store

class VoiceId(BaseModel):
    id: str


class DesignRequest(BaseModel):
    name: str = ""
    language: Literal["zh", "en"]
    instruction: str = Field(min_length=1, max_length=1000)
    text: str | None = None
    seed: int | None = None


@app.get("/voices")
async def list_voices():
    return {"voices": app.state.voices.list()}


@app.post("/voices/upload")
async def upload_voice(request: Request, name: str = "", language: str = "", text: str = ""):
    if language not in ("", "zh", "en"):
        raise HTTPException(400, "language 只能是 zh 或 en")
    with tempfile.NamedTemporaryFile(dir=app.state.data_dir, suffix=".upload") as upload:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "音频文件不能超过 30 MB")
            upload.write(chunk)
        upload.flush()
        try:
            pcm = await asyncio.to_thread(decode, upload.name, 24000, 20)  # the first 20 s
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if len(pcm) < 3 * 24000:
            raise HTTPException(400, "音频太短，至少需要 3 秒")
        text = text.strip()
        if not text:
            try:
                text, _ = await app.state.asr.submit(await asyncio.to_thread(decode, upload.name, 16000, 20))
            except Exception as exc:
                raise HTTPException(500, f"自动转写失败：{exc}")
            if not text:
                raise HTTPException(400, "没有识别出文字，请手动填写参考文本")
    return app.state.voices.add(name.strip() or "上传的音色", language or language_of(text), "upload", pcm, text)


@app.post("/voices/design")
async def design_voice(request: DesignRequest):
    breeze = app.state.tts.breeze
    if not breeze.configured:
        raise HTTPException(400, "音色设计需要 Breeze：请先填写 Breeze 的 Python、代码目录和模型目录")
    text = (request.text or "").strip() or DESIGN_TEXT[request.language]
    seed = random.randrange(2 ** 31) if request.seed is None else request.seed
    try:
        wav = await breeze.design(request.instruction.strip(), text, seed)
    except TTSError as exc:
        raise HTTPException(502, str(exc))
    audio, rate = sf.read(io.BytesIO(wav), dtype="int16")
    duration = len(audio) / rate
    if not 2 <= duration <= 30:
        raise HTTPException(422, f"生成的音频时长 {duration:.1f} 秒，不在 2–30 秒之间，请换个描述或种子再试")
    if np.abs(audio).max() < 33:  # about -60 dBFS peak
        raise HTTPException(422, "生成的音频是静音，请换个描述或种子再试")
    return app.state.voices.add(request.name.strip() or request.instruction.strip()[:20], request.language,
                                "design", audio, text, request.instruction.strip())


@app.post("/voices/delete")
async def delete_voice(request: VoiceId):
    try:
        app.state.voices.delete(request.id)
    except KeyError as exc:
        raise HTTPException(404, exc.args[0])
    return {"ok": True}


@app.get("/voices/audio")
async def voice_audio(id: str):
    try:
        return FileResponse(app.state.voices.dir(id) / "ref.wav", media_type="audio/wav")
    except KeyError as exc:
        raise HTTPException(404, exc.args[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    TOKEN = os.environ.get("RV_TOKEN", "")  # from the environment, never argv
    if not TOKEN:
        parser.error("RV_TOKEN is not set")
    app.state.data_dir = Path(args.data_dir)
    app.state.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=args.port, timeout_graceful_shutdown=5, access_log=False)
