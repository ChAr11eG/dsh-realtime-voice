"""TTS providers (breeze, edge, openai). Each streams PCM16LE 24 kHz mono in even-length chunks."""

import asyncio
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading

import httpx

from voices import FFMPEG

ENGINE = Path(__file__).resolve().parent


class TTSError(Exception):
    pass


def language_of(text):
    return "zh" if re.search(r"[㐀-鿿]", text) else "en"


async def even(chunks):
    """Carry an odd trailing byte so no sample is split across chunks."""
    carry = b""
    async for chunk in chunks:
        chunk = carry + chunk
        cut = len(chunk) - len(chunk) % 2
        carry = chunk[cut:]
        if cut:
            yield chunk[:cut]


async def ffmpeg_pcm(source, input_format=None):
    """Decode an encoded byte stream to PCM16 24 kHz mono while it arrives."""
    fmt = ["-f", input_format] if input_format else []
    process = await asyncio.create_subprocess_exec(
        FFMPEG, "-loglevel", "error", "-probesize", "32", "-analyzeduration", "0", *fmt, "-i", "pipe:0",
        "-f", "s16le", "-ar", "24000", "-ac", "1", "pipe:1",
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    async def feed():
        try:
            async for chunk in source:
                process.stdin.write(chunk)
                await process.stdin.drain()
        finally:
            process.stdin.close()

    writer = asyncio.create_task(feed())
    try:
        while chunk := await process.stdout.read(9600):
            yield chunk
        await writer  # raises the source's error, if any
        if await process.wait() != 0:
            raise TTSError("ffmpeg 解码失败：" + (await process.stderr.read()).decode(errors="replace")[-300:])
    finally:
        writer.cancel()
        if process.returncode is None:
            process.kill()
            await process.wait()


class Breeze:
    """Supervises the Breeze worker (engine/breeze_worker.py in Breeze's own venv)."""

    def __init__(self, data_dir, client):
        self.log = Path(data_dir) / "breeze.log"
        self.client = client
        self.cfg, self.key = {}, None
        self.process, self.task = None, None
        self.state, self.error = "off", None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.cfg.get('port') or 47911}"

    @property
    def configured(self):
        return all(self.cfg.get(name) for name in ("python", "codeDir", "model"))

    def configure(self, cfg, active):
        key = json.dumps([cfg.get(name) for name in ("python", "codeDir", "model", "port")])
        if key != self.key or not active:
            self.stop()  # frees the GPU when Breeze is no longer selected or its setup changed
        self.cfg, self.key = cfg, key

    def health(self):
        if self.state == "ready" and self.process is not None and self.process.poll() is not None:
            self.state, self.error = "error", f"Breeze 进程已退出（代码 {self.process.returncode}），见 {self.log}"
        return {"state": self.state, "error": self.error}

    async def start(self):
        if self.health()["state"] == "ready":
            return
        if self.task is None or self.task.done():
            self.state, self.error = "starting", None
            self.task = asyncio.create_task(self._run())
        await asyncio.shield(self.task)

    async def _run(self):
        try:
            if not self.configured:
                raise TTSError("请先填写 Breeze 的 Python、代码目录和模型目录")
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", int(self.cfg.get("port") or 47911))) == 0:
                    raise TTSError(f"Breeze 端口 {self.cfg.get('port')} 已被占用")
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}
            with open(self.log, "ab") as log:
                # Same process group as the engine, so stopping the group stops the worker too.
                self.process = subprocess.Popen(
                    [self.cfg["python"], "-u", str(ENGINE / "breeze_worker.py"), "--port", str(self.cfg.get("port") or 47911),
                     "--model", self.cfg["model"], "--code-dir", self.cfg["codeDir"]],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, cwd=ENGINE.parent)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 300
            while loop.time() < deadline:
                if self.process.poll() is not None:
                    tail = self.log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
                    raise TTSError(f"Breeze 进程退出（代码 {self.process.returncode}）：{tail[0][-300:]}")
                try:
                    if (await self.client.get(self.url + "/health", timeout=2)).status_code == 200:
                        self.state = "ready"
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
            raise TTSError("Breeze 启动超时（300 秒）")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.stop()
            self.state, self.error = "error", str(exc)
            raise

    def stop(self, wait=False):
        if self.task is not None and not self.task.done() and self.task is not asyncio.current_task():
            self.task.cancel()
        process, self.process = self.process, None
        self.state, self.error = "off", None
        if process is None or process.poll() is not None:
            return

        def reap():
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        process.terminate()
        if wait:
            reap()
        else:
            threading.Thread(target=reap, daemon=True).start()

    async def synthesize(self, text, language, ref_audio, ref_text):
        await self.start()
        body = {"text": text, "language": language, "ref_audio": ref_audio, "ref_text": ref_text}
        async with self.client.stream("POST", self.url + "/synthesize", json=body,
                                      timeout=httpx.Timeout(120, connect=5)) as response:
            if response.status_code != 200:
                raise TTSError(f"Breeze 合成失败（{response.status_code}）：{(await response.aread()).decode(errors='replace')[:300]}")
            async for chunk in even(response.aiter_raw()):
                yield chunk

    async def design(self, instruction, text, seed):
        await self.start()
        response = await self.client.post(self.url + "/design", timeout=600, json={
            "instruction": instruction, "text": text, "seed": seed, "cfg": 4.0})
        if response.status_code != 200:
            raise TTSError(f"Breeze 音色设计失败（{response.status_code}）：{response.text[:300]}")
        return response.content


class TTS:
    def __init__(self, data_dir, voices):
        self.voices = voices
        self.client = httpx.AsyncClient()
        self.breeze = Breeze(data_dir, self.client)
        self.cfg = {}
        self.edge_voices = None

    def configure(self, cfg):
        self.cfg = cfg
        self.breeze.configure(cfg.get("breeze") or {}, cfg["provider"] == "breeze")

    def health(self):
        if self.cfg["provider"] != "breeze":
            return {"provider": self.cfg["provider"], "ready": True, "loading": False, "error": None}
        breeze = self.breeze.health()
        return {"provider": "breeze", "ready": breeze["state"] == "ready", "loading": breeze["state"] == "starting",
                "error": breeze["error"]}

    async def warm(self):
        if self.cfg["provider"] == "breeze":
            await self.breeze.start()

    def stream(self, text, language=None, provider=None, voice=None):
        provider = provider or self.cfg["provider"]
        cfg = self.cfg.get(provider)
        if provider == "breeze":
            voice_id = voice or cfg.get("voiceId")
            if not voice_id:
                raise TTSError("请先在音色库中选择一个 Breeze 音色")
            ref_audio, ref_text = self.voices.reference(voice_id)
            return self.breeze.synthesize(text, language or language_of(text), ref_audio, ref_text)
        if provider == "edge":
            return even(ffmpeg_pcm(self._edge(text, voice or cfg.get("voice") or "zh-CN-YunxiNeural",
                                              cfg.get("rate") or "+0%", cfg.get("pitch") or "+0Hz"), "mp3"))
        if provider == "openai":
            return self._openai(text, voice or cfg.get("voice") or "alloy", cfg)
        raise TTSError(f"未知的语音合成服务：{provider}")

    async def _edge(self, text, voice, rate, pitch):
        import edge_tts
        async for item in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
            if item["type"] == "audio":
                yield item["data"]

    async def _openai(self, text, voice, cfg):
        headers = {"Authorization": f"Bearer {cfg['apiKey']}"} if cfg.get("apiKey") else {}
        body = {"model": cfg.get("model") or "gpt-4o-mini-tts", "voice": voice, "input": text, "response_format": "pcm"}
        async with self.client.stream("POST", cfg.get("baseUrl", "").rstrip("/") + "/audio/speech", json=body,
                                      headers=headers, timeout=httpx.Timeout(120, connect=10)) as response:
            if response.status_code >= 400:
                raise TTSError(f"云端合成失败（{response.status_code}）：{(await response.aread()).decode(errors='replace')[:300]}")
            kind = response.headers.get("content-type", "")
            raw = "pcm" in kind or "octet-stream" in kind or "l16" in kind
            async for chunk in even(response.aiter_raw() if raw else ffmpeg_pcm(response.aiter_raw())):
                yield chunk

    async def edge_catalog(self):
        if self.edge_voices is None:
            import edge_tts
            self.edge_voices = [
                {"id": voice["ShortName"], "language": voice["Locale"], "gender": voice["Gender"],
                 "name": f"{voice['ShortName'].split('-')[-1].removesuffix('Neural')} ({voice['Locale']})"}
                for voice in await edge_tts.list_voices() if voice["Locale"].startswith(("zh-", "en-"))]
        return self.edge_voices

    async def close(self):
        self.breeze.stop(wait=True)
        await self.client.aclose()
