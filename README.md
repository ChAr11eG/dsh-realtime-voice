# dsh-realtime-voice

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (DSH) plugin for a
realtime voice phone call with an agent persona, 大肥鱼 by default. Click **打电话给大肥鱼** in
the DSH sidebar, or ask in chat ("打电话给大肥鱼，聊聊周末去哪玩"), and a call screen opens
inside DSH. You talk through the browser microphone; the persona answers in a voice you
choose, and you can interrupt it mid-sentence.

The realtime line: Silero VAD with end-of-turn rules (a shorter wait after a question),
Qwen3-ASR with a CUDA-graph decoder and live partial transcripts, phrase-by-phrase streaming
TTS, and a three-word barge-in rule. Echo cancellation is left to the browser.

Tested with DSH 0.2.2-rc.1 on Linux (Chromium, RTX 4090): about 0.45 s from the end of
speech to the final transcript and 0.2–0.3 s from the transcript to the first reply audio
with Breeze TTS and a local 9B model.

## How it fits together

```
DSH page (client.js) ──WS /realtime-voice/ws, HTTP /api/realtime-voice/*──▶ host plugin (Node, inside DSH)
                                                                              │  LLM: DSH's configured models
                                                                              │       or any OpenAI-compatible API
                                                                              ▼
                                                           speech engine (Python, 127.0.0.1)
                                                           VAD + ASR + TTS + voice library
                                                                              ▼
                                                           Breeze TTS 2 worker (optional, own venv)
```

- **client.js** adds the settings page (设置 → 实时语音), the call page, the sidebar entry,
  a button on the new-session screen, and a card for the chat tool. The call page runs in
  DSH's own page, because DSH's in-app browser and iframes cannot use the microphone.
- **host/** owns the conversation: LLM streaming, phrase splitting, barge-in, call records,
  and the Python engine process (started on demand, stopped after 15 idle minutes).
  It registers the tool `realtime_voice_call`, so DSH's agent can open a call for you.
- **engine/** owns the audio: turn-taking, speech recognition, speech synthesis and voices.

[SPEC.md](SPEC.md) is the full contract between the three parts.

## Providers

| | Options |
|---|---|
| Speech recognition | **Qwen3-ASR 0.6B** (local, CUDA, live partials), **OpenAI-compatible** `/audio/transcriptions` (Whisper, Groq, SiliconFlow, …) |
| Language model | **DSH's configured models** (the same providers as chat; thinking turned off), **OpenAI-compatible** `/chat/completions` (DeepSeek API, llama.cpp, …) |
| Speech synthesis | **Edge** (free online Microsoft voices, no GPU), **Breeze TTS 2** (local GPU, clone any voice from a 3–20 s clip or design one from a description), **OpenAI-compatible** `/audio/speech` |

## Install

Requirements: Node ≥ 24.9, Python 3.10+, `ffmpeg`. Local ASR and Breeze need an NVIDIA GPU.

1. Engine environment (creates `.venv`):

   ```bash
   scripts/setup-python.sh
   ```

   To reuse an environment that already has torch and qwen-asr (no multi-GB download):

   ```bash
   BASE_PYTHON=/path/to/env/bin/python scripts/setup-python.sh
   ```

2. Optional, for local voice cloning (downloads about 8 GB into `.breeze/`):

   ```bash
   scripts/setup-breeze.sh
   ```

3. Node dependencies:

   ```bash
   npm install
   ```

4. Add the plugin to DSH. Either install it into the web profile (needs `pnpm` on PATH):

   ```bash
   npx @deepseek-ai/dsh plugin --profile web add link:$PWD
   ```

   or run DSH with this checkout as an overlay, without installing anything:

   ```bash
   npx @deepseek-ai/dsh web --patch ./dev.patch.yml
   ```

   Restart DSH after installing or updating, then reload the page.

## First call

1. Open 设置 → 实时语音.
2. 语言模型: keep "DSH 已配置的模型" and pick a provider and model, or switch to an
   OpenAI-compatible endpoint and paste its key.
3. 语音识别: Qwen3-ASR downloads `Qwen/Qwen3-ASR-0.6B` on first use, or point it at a local
   copy. Choose OpenAI-compatible instead if there is no GPU.
4. 语音合成: Edge works out of the box. For Breeze, fill in the three paths printed by
   `setup-breeze.sh`, add a voice in 音色库 (upload a clip, or describe a voice), and select it.
5. Click **打电话给大肥鱼** in the sidebar, then the green button. The first call waits while
   the models load (about 20 s with Breeze).

Settings are saved when a field loses focus. API keys are stored in
`~/.dsh/realtime-voice/secrets.json` (mode 0600) and are never sent back to the page.
Call transcripts are kept in `~/.dsh/realtime-voice/calls/` and listed under 通话记录.

## Notes and limits

- Microphone access needs a secure page: `http://127.0.0.1` / `localhost` works; another
  device needs DSH started with TLS.
- Tested on DSH Web only. The desktop app should work in principle (the call page runs in the
  main window, which is allowed the microphone), but its WebSocket origin is untested.
- In DSH 0.2.2-rc.1 the first click on any sidebar panel right after a page load only
  highlights it (the built-in 自动化任务 panel does the same); click again.
- One call at a time. A new call starts a new conversation; the persona prompt and the topic
  passed by the chat tool are its only context.
- Breeze TTS 2 weights are for research and non-commercial use. Only clone voices you have
  permission to use.
- GPU memory with everything local: Qwen3-ASR about 2–3 GiB, Breeze about 12 GiB.

## Development

```bash
DSH_HOME=/tmp/dsh-dev npx @deepseek-ai/dsh web --patch ./dev.patch.yml --no-open --port 3099
```

Open the printed `?token=` URL once. Host-side changes need a DSH restart; client changes
need a page reload as well. The engine log is at `~/.dsh/realtime-voice/engine.log` and the
last lines are shown under 引擎 → 引擎日志.

## License

MIT for this repository. Model weights and third-party packages keep their own licenses.
