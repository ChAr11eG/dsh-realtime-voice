# dsh-realtime-voice: build contract

A DeepSeek Harness (DSH 0.2.2-rc.1) plugin for a realtime voice "phone call" with an agent
persona (default name 大肥鱼). Echo cancellation is left to the browser
(`echoCancellation`).

Three parts, three owners. Each part may only rely on the interfaces in this file.

```
browser (DSH page, client.js) ──WS /realtime-voice/ws──▶ host plugin (Node, in the DSH process)
                               ──HTTP /api/realtime-voice/*──▶      │
                                                                     │ HTTP + WS, 127.0.0.1, token
                                                                     ▼
                                                    speech engine (Python, engine/server.py)
                                                       VAD + ASR + TTS + voice store
                                                                     │ HTTP 127.0.0.1
                                                                     ▼
                                                    Breeze worker (engine/breeze_worker.py,
                                                    runs in Breeze's own venv; optional)
```

The host owns the conversation: LLM streaming, phrase splitting, barge-in, turn counter,
transcripts. The engine owns audio intelligence: turn-taking (VAD, end of turn), ASR, TTS.

## Repository layout

```
package.json  cordis.bundle.yml  dev.patch.yml  icon.svg  locale/en.json  locale/zh.json
dsh-plugin.mjs          host entry (named exports only, no default export)
host/settings.mjs       settings.json + secrets.json store
host/engine.mjs         Python engine supervisor and HTTP client
host/llm.mjs            LLM providers: "dsh" (ctx.llm) and "openai" (OpenAI-compatible)
host/text.mjs           firstPhrase, nextPhrase, substantial, cleanSpeech, language
host/call.mjs           one call session (port of server.py Session)
host/routes.mjs         /api/realtime-voice/* routes and the WS upgrade
client.js               browser bundle (hand-written window.__ModuleLoader__ factory)
engine/server.py        FastAPI engine
engine/vad.py           Silero VAD and the turn segmenter
engine/asr.py           ASR providers: qwen3 (local), openai (cloud)
engine/asr_graph.py     CUDA-graph decoder for Qwen3-ASR
engine/tts.py           TTS providers: breeze, edge, openai
engine/voices.py        voice store
engine/breeze_worker.py Breeze TTS 2 worker (separate venv)
engine/requirements.txt
scripts/setup-python.sh
README.md
```

## 1. Engine (Python) — `engine/server.py`

Started by the host: `RV_TOKEN=<token> <python> -u engine/server.py --port <port> --data-dir <dir>`
with cwd = repository root. Binds 127.0.0.1 only. Every HTTP request must carry header
`x-rv-token: <token>`, the WebSocket included; otherwise 401 / close 4401. The token never
appears in argv or logs (uvicorn access log off).
Exits cleanly on SIGTERM and terminates the Breeze worker it started.

All JSON errors: HTTP 4xx/5xx with `{"detail": "<human readable, Chinese ok>"}`.

### Engine config (POST /config body; the host always sends the whole object)

```json
{
  "asr": {
    "provider": "qwen3" | "openai",
    "language": "auto" | "zh" | "en",
    "partials": true,
    "partialInterval": 0.25,
    "qwen3": { "model": "Qwen/Qwen3-ASR-0.6B", "device": "cuda" },
    "openai": { "baseUrl": "https://api.openai.com/v1", "apiKey": "", "model": "whisper-1" }
  },
  "turn": {
    "vadThreshold": 0.5, "silence": 0.4, "confirm": 0.16,
    "questionSilence": 0.2, "questionConfirm": 0.05,
    "minSpeech": 0.35, "preRoll": 0.6, "earlyTranscribe": 0.2
  },
  "tts": {
    "provider": "breeze" | "edge" | "openai",
    "breeze": { "python": "", "codeDir": "", "model": "", "voiceId": "", "port": 47911 },
    "edge": { "voice": "zh-CN-YunxiNeural", "rate": "+0%", "pitch": "+0Hz" },
    "openai": { "baseUrl": "https://api.openai.com/v1", "apiKey": "", "model": "gpt-4o-mini-tts", "voice": "alloy" }
  }
}
```

`qwen3.model` is a local directory or a Hugging Face id. Applying a config never blocks on
model loading: changed providers are marked not ready and loaded by `/warm` or lazily on
first use.

### Endpoints

- `GET /health` → `{"status":"ok","asr":{"provider","ready":bool,"loading":bool,"error":str|null},
  "tts":{"provider","ready","loading","error"},"breeze":{"state":"off"|"starting"|"ready"|"error","error"}}`
- `POST /config` → applies, returns `/health` body.
- `POST /warm` → loads the selected ASR and TTS providers now (Qwen3 weights + CUDA graph
  capture + one dummy transcription; starts the Breeze worker and waits for its `/health`,
  up to 300 s). Returns `/health` body; 500 with `detail` if a provider failed.
- `WS /asr` (header `x-rv-token`) — one utterance stream per connection (one connection at a time; a
  second gets close 4409).
  - Client → engine: text `{"type":"start"}` first (optional), then binary frames: PCM16LE,
    16 kHz, mono, any even length 2–6400 bytes (the host forwards 20 ms = 640-byte frames).
    Text `{"type":"reset"}` drops the current utterance, pre-roll and any pending ASR.
  - Engine → client (text JSON):
    - `{"type":"speech_start"}` — VAD onset (speech probability > `vadThreshold` for 2
      consecutive 32 ms Silero frames). The pre-roll (`preRoll` s of audio before onset) is
      kept with the utterance.
    - `{"type":"partial","text"}` — while speaking, from 0.3 s after onset, every
      `partialInterval` s, the whole utterance so far is transcribed (skipped while the ASR
      is busy; only when `asr.partials` is true). Empty texts are not sent.
    - `{"type":"final","text","asr_seconds"}` — end of turn. End of turn = continuous
      non-speech for `confirm + silence` s, or `questionConfirm + questionSilence` s when the
      latest partial ends with `?` or `？`. Early transcription: after `earlyTranscribe` s of
      silence the final ASR starts in the background; if speech resumes it is discarded.
      An utterance shorter than `minSpeech` s from onset, or an empty transcript, yields
      `{"type":"no_speech"}` instead.
    - `{"type":"error","detail"}`.
  - The ASR runs one request at a time (one worker thread + lock). Finals have priority over
    partials.
- `POST /tts` body `{"text": str (1–2000), "language": "zh"|"en"|null, "provider": str|null,
  "voice": str|null}` → `200 application/octet-stream`, header `X-Sample-Rate: 24000`, body
  streamed PCM16LE 24 kHz mono, chunks forwarded as soon as produced, every chunk an even
  number of bytes. `provider`/`voice` override the config for previews (voice = Breeze voice
  id, Edge voice name, or OpenAI voice name). Closing the connection cancels synthesis.
  Errors before the first byte → JSON error status.
- `GET /tts/voices?provider=edge` → `{"voices":[{"id","name","language","gender"}]}` (Edge
  catalog, zh-* and en-* only, cached). Other providers → `{"voices":[]}`.
- Voice store (Breeze clone voices):
  - `GET /voices` → `{"voices":[Voice]}` where
    `Voice = {"id","name","language":"zh"|"en","source":"upload"|"design","duration":float,
    "text","created":ISO8601,"instruction":str|null}`.
  - `POST /voices/upload?name=&language=&text=` body = raw audio bytes (any format ffmpeg
    can decode, max 30 MB) → decoded to 24 kHz mono PCM16, longest 20 s kept (first 20 s),
    at least 3 s required. If `text` is empty it is transcribed with the ASR provider (must
    be ready/loadable) → returns `Voice`.
  - `POST /voices/design` body `{"name","language","instruction","text":str|null,
    "seed":int|null}` → Breeze voice design (CFG 4, instruction only), output must be 2–30 s
    and non-silent, saved as a clone reference → `Voice`. Requires Breeze configured.
  - `POST /voices/delete` body `{"id"}` → `{"ok":true}`.
  - `GET /voices/audio?id=` → `audio/wav` of the reference.
  - Storage: `<data-dir>/voices/<id>/{ref.wav,ref.txt,meta.json}`, id = short random hex.

### Providers

- ASR `qwen3`: `qwen_asr` package (as used by RealtimeSTT's `qwen3_asr_engine.py`), bf16 on
  CUDA, `max_new_tokens=128`, with `asr_graph.install()` (CUDA-graph decode) before loading.
- ASR `openai`: `POST {baseUrl}/audio/transcriptions` multipart `file` (16 kHz WAV), `model`,
  optional `language`; `Authorization: Bearer <apiKey>` when set. Partials default off for
  cloud (the host sets `asr.partials` from settings).
- TTS `breeze`: proxy to the Breeze worker with the selected voice's `ref.wav` + `ref.txt`;
  language from the request, else Han characters → zh, else en.
- TTS `edge`: `edge_tts.Communicate(text, voice, rate=, pitch=)` MP3 stream piped through
  `ffmpeg -f mp3 -i pipe:0 -f s16le -ar 24000 -ac 1 pipe:1`, streamed.
- TTS `openai`: `POST {baseUrl}/audio/speech` `{model, voice, input, response_format:"pcm"}`;
  if the reply is not raw PCM (content-type audio/mpeg, audio/wav, …) decode through ffmpeg
  to 24 kHz PCM16.

### Breeze worker — `engine/breeze_worker.py`

Runs under `tts.breeze.python` (Breeze's own venv: torch 2.9.1, transformers 4.57.3) with
`--port --model <dir> --code-dir <breeze-tts checkout>`; started by the engine in the same
process group, 127.0.0.1 only. Model/runtime setup:
(FastBreezeStreamingRuntime, CUDA-graph warmup from a clone-only profile, seed 42, CFG 1,
`MAX_NEW_TOKENS=500`, `MAX_SEQ_LEN=1024`, one lock, generation in its own thread, client
disconnect cancels). Endpoints: `GET /health`; `POST /synthesize {text, ref_audio, ref_text,
language}` → streamed PCM16LE 24 kHz (encoded references cached by path+mtime);
`POST /design {instruction, text, seed, cfg}` → `audio/wav` (separate un-warmed eager runtime,
`max_new_tokens=1500`, `max_seq_len=2048`).

## 2. Host plugin (Node) — `dsh-plugin.mjs` + `host/*`

Plain ESM, no `@deepseek-ai/*` imports, Node ≥ 24.9. Dependency: `ws`.
`export const name = 'realtime-voice'`, `export const inject = ['tools']`,
`export async function apply(ctx, config = {})`. `webServer`, `connection` and `llm` are
reached with `ctx.inject([...], sub => …)` / `ctx.get('llm')`. Everything unwinds via
disposers / `ctx.effect`.

Data dir: `config.dataDir`, else `$DSH_HOME/realtime-voice`, else `~/.dsh/realtime-voice`.

### Settings (`settings.json`, atomic write) — defaults

```js
{
  persona: { name: '大肥鱼', prompt: '<default persona, Chinese>', greeting: '喂？我是大肥鱼，找我什么事？' },
  llm: { provider: 'dsh', dsh: { provider: '', model: '' },
         openai: { baseUrl: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
         maxTokens: 300, temperature: 0.7 },
  asr: { provider: 'qwen3', language: 'auto', partials: true, partialInterval: 0.25,
         qwen3: { model: 'Qwen/Qwen3-ASR-0.6B', device: 'cuda' },
         openai: { baseUrl: 'https://api.openai.com/v1', model: 'whisper-1' } },
  turn: { vadThreshold: 0.5, silence: 0.4, confirm: 0.16, questionSilence: 0.2,
          questionConfirm: 0.05, minSpeech: 0.35, preRoll: 0.6, earlyTranscribe: 0.2 },
  tts: { provider: 'edge',
         breeze: { python: '', codeDir: '', model: '', voiceId: '', port: 47911 },
         edge: { voice: 'zh-CN-YunxiNeural', rate: '+0%', pitch: '+0Hz' },
         openai: { baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o-mini-tts', voice: 'alloy' } },
  engine: { python: '', port: 47910, idleMinutes: 15 },
  bargeIn: { minWords: 3 }
}
```

Secrets (`secrets.json`, mode 0600, never returned): `llmOpenaiKey`, `asrOpenaiKey`,
`ttsOpenaiKey`. Engine python default: `<repo>/.venv/bin/python` if it exists, else
`python3`.

### HTTP API (`/api/realtime-voice/*`, JSON, same-origin browser only)

Registered as one `webServer` prefix route `/api/realtime-voice`; every request is checked
with `connection.requestRejection(req)` (401/403 → reject) and bodies are size-limited
(JSON 256 KB, uploads 30 MB). Paths (query strings allowed):

| Method + path | Body | Reply |
|---|---|---|
| GET `/state` | – | `State` |
| POST `/settings` | partial settings (deep merge, validated) | `State` |
| POST `/secrets` | `{name, value}` (`''` clears) | `State` |
| POST `/engine/start` | – | `State` (returns once the process answers `/health`; warming continues) |
| POST `/engine/stop` | – | `State` |
| GET `/voices` | – | engine `/voices` |
| POST `/voices/upload?name=&language=&text=` | raw audio | engine reply |
| POST `/voices/design` | JSON | engine reply |
| POST `/voices/delete` | `{id}` | engine reply |
| GET `/voices/audio?id=` | – | `audio/wav` |
| GET `/tts/voices?provider=edge` | – | engine reply |
| POST `/tts/preview` | `{text, provider?, voice?}` | `audio/wav` (24 kHz) |
| GET `/calls` | – | `{calls:[{id, started, ended, persona, topic, turns, preview}]}` newest first, max 50 |
| GET `/calls/get?id=` | – | full call record |

Voice/TTS routes start the engine on demand (and warm it when needed).

`State = { settings, secrets: {llmOpenaiKey: bool, asrOpenaiKey: bool, ttsOpenaiKey: bool},
engine: { state: 'stopped'|'starting'|'warming'|'ready'|'error', error, health, log: [last 80 lines] },
llmCatalog: [{ id, name, models: [{ id, name }] }]  // from ctx.llm, [] if unavailable
, pendingCall: { topic, at } | null, inCall: bool, repoDir, dataDir }`

### Engine supervisor

Spawn with `detached: true` (own process group); stdout+stderr to `<dataDir>/engine.log`
and an 80-line ring buffer. Wait for `/health` (60 s), POST `/config`, then POST `/warm`
(state `warming` until it returns, `ready` after, `error` with detail on failure). Stop =
SIGTERM to the group, SIGKILL after 10 s. Stopped on plugin dispose. Idle stop after
`engine.idleMinutes` (0 = never) with no call. Settings changes push `/config` to a running
engine (and re-warm if a provider changed).

### LLM (`host/llm.mjs`)

`stream({system, messages:[{role:'user'|'assistant', text}], signal}) → AsyncIterable<string>`
- `dsh`: `ctx.get('llm').stream({provider, model, system, messages, maxTokens, temperature,
  signal, reasoningEffort:'off' when the model lists an 'off' effort})`; user message
  `{role:'user', content:[{type:'text', text}]}`, assistant
  `{id, role:'assistant', content:[{type:'text', text}], source:{kind:'model', provider, model}}`;
  yield `text-delta` text; `finish` with kind `error` → throw. Empty `dsh.provider/model`
  → first configured provider/model.
- `openai`: `POST {baseUrl}/chat/completions` with `stream: true`, SSE parse, Bearer key.
- `catalog()` → `[{id, name, models:[{id,name}]}]` (5 s timeout, cached 30 s).

System prompt = persona.prompt + fixed voice rules (spoken Chinese/English matching the
user, 1–3 short sentences, no markdown/lists/emoji/URLs, numbers in words, the persona name)
+ `用户想聊的话题：<topic>` when a topic is pending.

### Text helpers (`host/text.mjs`)

- `firstPhrase(text)` — first phrase to synthesize (split tokens
  `. ! ? , ; : \n - 。 、 ， ！ ？ ； ：`, weight: alnum 1, Han 2.5, min 10, max 120; ASCII
  punctuation only before whitespace).
- `nextPhrase(text)` — later phrases: split at `. ! ? ; \n 。 ！ ？ ；` with weight ≥ 10, or at
  `, ， 、 ：` with weight ≥ 40, or force at 200 chars; same ASCII rule.
- `substantial(text, minWords)` — ≥ minWords Han chars or ≥ minWords `[A-Za-z0-9']+` words.
- `cleanSpeech(text)` — drop `<think>…</think>`, markdown `* _ # \``, emoji, URLs; collapse
  whitespace.
- `language(text)` — `zh` if any Han char else `en`.

### Call session (`host/call.mjs`)

One call at a time (a second WS gets `{"type":"error","detail":"已经在通话中"}` and close
1008). On connect: send `status` messages while the engine starts/warms, open the engine
`/asr` WS, send `ready`, then speak the greeting (if non-empty) as the first assistant turn.

Barge-in (`minWords = settings.bargeIn.minWords`):
- `playing` = a reply is being generated or its audio has not been acknowledged by
  `playback_done`.
- engine `speech_start` while playing → `echoSuspect = true`, forward `speech_start`. If
  `minWords == 0`, interrupt immediately instead.
- `partial` while `echoSuspect && playing` and `substantial` → forward `partial_user`, interrupt.
- `final` while `echoSuspect` and not substantial → `ignored_user`, no reply.
- `speech_start` while not playing → forward. `final` (non-empty) → forward `final_user`,
  interrupt anything playing, start a reply. `no_speech` → clear echoSuspect.
- interrupt = `turn += 1`, send `{"type":"stop_audio","turn"}` FIRST, abort the LLM and the
  in-flight TTS fetch; anything tagged with an older turn is dropped.
- client `{"type":"interrupt"}` = interrupt now.

Reply: append user text to history (last 20 messages kept), stream the LLM, cut with
`firstPhrase` then `nextPhrase`, `cleanSpeech` each phrase, synthesize phrases sequentially
through engine `/tts` (language per phrase), forward PCM to the browser as it arrives (even
byte boundaries), and send `assistant_delta` with each phrase's text when its first audio
byte is sent. A reply is capped at 90 s. On completion append the spoken text to history,
send `final_assistant` and `audio_end`. If interrupted, the spoken part so far is stored as
the assistant message.

On close: abort everything, close the engine WS, save `<dataDir>/calls/<id>.json`
`{id, started, ended, persona, topic, messages:[{role, text, at}]}`, clear `pendingCall`,
start the idle timer.

### Tool

`realtime_voice_call` — description (Chinese + English) tells the model to call it when the
user wants to talk by voice / phone the persona (e.g. "打电话给大肥鱼"). Params
`{topic?: string}` (no min/max/pattern keywords). Sets `pendingCall = {topic, at}` and
returns `{ok: true, persona, topic}`; `render` → one text block telling the model the call
page is open in DSH and the user can talk now. `timeoutMs: 10000`.

## 3. Call WebSocket protocol (`/realtime-voice/ws`)

Registered with `webServer.registerUpgrade({path:'/realtime-voice/ws'})`, guarded by
`connection.requestRejection(req)` (reject → write `HTTP/1.1 401|403` and close).

Browser → host:
- binary: PCM16LE 16 kHz mono, 20 ms frames (640 bytes; accept even lengths 2–6400).
- `{"type":"hello","client":"dsh-realtime-voice"}` (first message, optional)
- `{"type":"interrupt"}`, `{"type":"playback_done","response_id"}`, `{"type":"hangup"}`

Host → browser:
- `{"type":"status","state":"starting"|"warming"|"ready","detail"}`
- `{"type":"ready","call_id","input_rate":16000,"output_rate":24000,"persona":{"name"},"topic"}`
- `{"type":"speech_start"}`, `{"type":"partial_user","text"}`, `{"type":"final_user","text"}`,
  `{"type":"ignored_user","text"}`
- `{"type":"stop_audio","turn"}`
- `{"type":"audio_start","turn","response_id","sample_rate":24000}`, then binary PCM16LE
  24 kHz mono, `{"type":"assistant_delta","response_id","text"}`,
  `{"type":"final_assistant","response_id","text","interrupted":bool}`,
  `{"type":"audio_end","response_id","samples","first_audio_seconds","generation_seconds"}`
- `{"type":"error","detail","scope"?}`, `{"type":"ended","reason"}`

## 4. Client (`client.js`)

`window.__ModuleLoader__.load({ id: 'dsh-realtime-voice', factory(require) {...} })`
returning `{ name: 'dsh-realtime-voice-client', inject: ['slots', 'layout'], apply(ctx) }`.
React via `require('react')` and `createElement` (no JSX, no build step); primitives from
`@deepseek-ai/dsh-client-ui-primitives` (Button, Switch, Input, SegmentedControl…); colours
only from `--dsw-alias-*` tokens. All UI text in Chinese.

Registrations (slots):
- `settings.section` id `realtime-voice`, label `实时语音`: the full settings page.
- `main` key `realtime-voice.call`: the call page. Open with
  `ctx.layout.selectPanel('realtime-voice.call')`.
- `sidebar.panellist` id `realtime-voice.call`, label `打电话给<persona name>`, phone icon.
- `conversation.hero.options`: a `📞 打电话给<name>` button.
- `tool.call.toolview` key `realtime_voice_call`: a call card with an "接通" button; auto-opens
  the call page once when the result is fresh (`block.time` within 30 s).
- optional `commandUi` (via `ctx.inject(['commandUi'], …)`) command `call`, label
  `打电话给<name>`.

The call engine lives in a module-level singleton (outside React) so switching panels does
not hang up. Capture: `getUserMedia({audio:{echoCancellation:true, noiseSuppression:true,
autoGainControl:true}})`, `AudioContext({sampleRate:16000})`, an AudioWorklet from a Blob
URL posting 320-sample Int16 frames. Playback: `AudioContext({sampleRate:24000})` + a
ring-buffer AudioWorklet; `stop_audio` flushes it; after `audio_end` and the buffer drains,
send `playback_done`. WS URL: same origin as the page (`ws(s)://location.host/realtime-voice/ws`).

Settings page sections: 角色 (name, prompt, greeting), 语言模型 (provider dsh/openai, DSH
provider+model dropdowns from `llmCatalog`, OpenAI-compatible base URL/model/key), 语音识别
(provider, Qwen3 model, cloud base URL/model/key, language, partials, end-of-turn timings),
语音合成 (provider, Edge voice dropdown + rate/pitch, OpenAI voice, Breeze paths + voice
picker, 试听 preview), 音色库 (list with play/delete, upload a clip with optional transcript,
design from a description), 引擎 (python path, port, idle minutes, state, start/stop, log),
通话记录 (recent calls). Text fields save on blur, selects/switches on change; secrets in
password fields, shown only as 已保存/未设置.
