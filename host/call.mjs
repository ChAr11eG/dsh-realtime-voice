// One call session: turn-taking, barge-in, LLM reply and phrase-by-phrase TTS.
import { randomBytes } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { errorDetail } from './engine.mjs'
import { systemPrompt } from './llm.mjs'
import { cleanSpeech, firstPhrase, language, nextPhrase, stripThink, substantial } from './text.mjs'

const HISTORY = 20
const REPLY_LIMIT_MS = 90_000

export class CallSession {
  constructor({ ws, engine, llm, settings, dataDir, topic }) {
    this.ws = ws
    this.engine = engine
    this.llm = llm
    this.settings = settings
    this.dataDir = dataDir
    this.topic = topic ?? ''
    this.id = `${new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}-${randomBytes(3).toString('hex')}`
    this.started = new Date().toISOString()
    this.persona = settings.get().persona.name
    this.history = [] // LLM context, last HISTORY messages
    this.record = [] // full transcript {role, text, at}
    this.turn = 0
    this.closed = false
    this.ready = false
    this.asr = null
    this.response = null // { ac, rid } while a reply is generated
    this.deliveryId = null // response_id until the browser acknowledges playback
    this.audioEnded = false
    this.echoSuspect = false // VAD onset while a reply was playing
    this.ended = new Promise(resolve => { this.resolveEnded = resolve })
  }

  get playing() {
    return this.response !== null || this.deliveryId !== null
  }

  get minWords() {
    return this.settings.get().bargeIn.minWords
  }

  send(data, turn) {
    if (this.closed || (turn !== undefined && turn !== this.turn)) return
    if (this.ws.readyState !== this.ws.OPEN) return
    this.ws.send(Buffer.isBuffer(data) ? data : JSON.stringify(data))
  }

  /** Start the engine, open the ASR stream, say hello. */
  async run() {
    this.ws.on('message', (data, isBinary) => this.onClient(data, isBinary))
    this.ws.on('close', () => this.close('hangup'))
    this.ws.on('error', () => this.close('error'))
    try {
      await this.engine.ensureReady((state, error) => {
        if (state === 'starting' || state === 'warming') this.send({ type: 'status', state, detail: error ?? '' })
      })
      if (this.closed) return
      await this.openAsr()
      if (this.closed) return
      this.ready = true
      this.send({ type: 'status', state: 'ready', detail: '' })
      this.send({
        type: 'ready', call_id: this.id, input_rate: 16000, output_rate: 24000,
        persona: { name: this.persona }, topic: this.topic,
      })
      const greeting = this.settings.get().persona.greeting.trim()
      if (greeting) this.startResponse(null, greeting)
    } catch (error) {
      this.send({ type: 'error', detail: error.message, scope: 'engine' })
      this.close('error')
    }
  }

  openAsr() {
    return new Promise((resolve, reject) => {
      const asr = this.asr = this.engine.asrSocket()
      asr.once('open', () => {
        asr.send(JSON.stringify({ type: 'start' }))
        resolve()
      })
      asr.once('error', reject)
      asr.on('message', (data, isBinary) => {
        if (!isBinary) this.onEngine(JSON.parse(data.toString()))
      })
      asr.on('close', code => {
        if (this.closed) return
        reject(new Error(`语音识别连接已断开（${code}）`))
        this.send({ type: 'error', detail: `语音识别连接已断开（${code}）`, scope: 'engine' })
        this.close('engine')
      })
    })
  }

  onClient(data, isBinary) {
    if (this.closed) return
    if (isBinary) {
      if (!data.length || data.length % 2 || data.length > 6400) {
        this.ws.close(1003, 'Expected 16 kHz mono PCM16, 1-200 ms')
        return
      }
      if (this.ready && this.asr.readyState === this.asr.OPEN) this.asr.send(data)
      return
    }
    let message
    try {
      message = JSON.parse(data.toString())
    } catch {
      return // not JSON: ignore
    }
    if (message.type === 'interrupt') this.interrupt()
    else if (message.type === 'playback_done') this.playbackDone(message.response_id)
    else if (message.type === 'hangup') this.close('hangup')
  }

  /** Turn-taking and barge-in. */
  onEngine(event) {
    if (this.closed) return
    const { type } = event
    const text = typeof event.text === 'string' ? event.text.trim() : ''
    if (type === 'speech_start') {
      if (this.playing && this.minWords > 0) {
        // A VAD onset alone does not stop playback: it may be our own echo.
        this.echoSuspect = true
      } else if (this.playing) {
        this.interrupt()
      }
      this.send({ type: 'speech_start' })
    } else if (type === 'partial') {
      if (this.echoSuspect && this.playing && substantial(text, this.minWords)) {
        this.echoSuspect = false
        this.send({ type: 'partial_user', text })
        this.interrupt()
      } else if (text) {
        this.send({ type: 'partial_user', text })
      }
    } else if (type === 'final') {
      if (this.echoSuspect && !substantial(text, this.minWords)) {
        this.echoSuspect = false
        this.send({ type: 'ignored_user', text })
      } else if (text) {
        this.echoSuspect = false
        this.send({ type: 'final_user', text })
        if (this.playing) this.interrupt()
        this.startResponse(text)
      } else {
        this.echoSuspect = false
      }
    } else if (type === 'no_speech') {
      this.echoSuspect = false
    } else if (type === 'error') {
      this.send({ type: 'error', detail: event.detail ?? '语音识别出错', scope: 'asr' })
    }
  }

  /** New turn: stop_audio first, then abort the LLM and the TTS fetch. */
  interrupt() {
    this.turn += 1
    this.deliveryId = null
    this.send({ type: 'stop_audio', turn: this.turn })
    this.abortResponse()
  }

  /** Abort the reply in flight and record what it already said, before anything newer is recorded. */
  abortResponse() {
    const response = this.response
    this.response = null
    if (!response) return
    response.ac.abort()
    response.recorded = true
    const answer = cleanSpeech(response.spoken)
    if (answer) this.pushHistory('assistant', answer)
  }

  playbackDone(rid) {
    if (rid && rid === this.deliveryId && this.audioEnded) {
      this.deliveryId = null
      this.audioEnded = false
    }
  }

  pushHistory(role, text) {
    this.history.push({ role, text })
    this.history.splice(0, Math.max(0, this.history.length - HISTORY))
    this.record.push({ role, text, at: new Date().toISOString() })
  }

  /** userText: reply to it with the LLM; fixedText: speak it as is (greeting). */
  startResponse(userText, fixedText) {
    const rid = randomBytes(8).toString('hex')
    const response = { ac: new AbortController(), rid, spoken: '', recorded: false }
    this.response = response
    this.deliveryId = rid
    this.audioEnded = false
    if (userText) this.pushHistory('user', userText)
    this.respond(response, this.turn, fixedText).finally(() => {
      if (this.response === response) this.response = null
    })
  }

  async respond(response, turn, fixedText) {
    const { ac, rid } = response
    const started = performance.now()
    const timer = setTimeout(() => ac.abort(new Error('回复超过 90 秒')), REPLY_LIMIT_MS)
    const phrases = []
    let llmDone = false
    let llmError = null
    let wake = null
    const notify = () => { wake?.(); wake = null }

    // Producer: stream the LLM and cut phrases as soon as they are long enough.
    const source = fixedText !== undefined
      ? (async function* () { yield fixedText })()
      : this.llm.stream({ system: systemPrompt(this.settings.get(), this.topic), messages: [...this.history], signal: ac.signal })
    const producer = (async () => {
      let raw = ''
      let taken = 0
      let split = firstPhrase
      try {
        for await (const delta of source) {
          if (ac.signal.aborted) break
          raw += delta
          const visible = stripThink(raw)
          let end
          while ((end = split(visible.slice(taken))) !== null) {
            phrases.push(visible.slice(taken, taken + end))
            taken += end
            split = nextPhrase
            notify()
          }
        }
        const rest = stripThink(raw).slice(taken)
        if (!ac.signal.aborted && rest.trim()) phrases.push(rest)
      } catch (error) {
        llmError = error
      } finally {
        llmDone = true
        notify()
      }
    })()

    let spokenAny = false
    let samples = 0
    let firstAudio = null
    try {
      for (;;) {
        if (ac.signal.aborted) break
        if (!phrases.length) {
          if (llmDone) break
          await new Promise(resolve => { wake = resolve })
          continue
        }
        const phrase = phrases.shift()
        const text = cleanSpeech(phrase)
        if (!text) continue
        spokenAny = true
        const res = await this.engine.request('POST', '/tts', {
          json: { text, language: language(text), provider: null, voice: null }, signal: ac.signal,
        })
        if (!res.ok) throw new Error(`语音合成失败：${await errorDetail(res)}`)
        let carry = null
        let phraseStarted = false
        for await (let chunk of res.body) {
          chunk = Buffer.from(chunk)
          if (carry) chunk = Buffer.concat([carry, chunk])
          carry = chunk.length % 2 ? chunk.subarray(chunk.length - 1) : null
          if (carry) chunk = chunk.subarray(0, chunk.length - 1)
          if (!chunk.length || turn !== this.turn) continue
          if (firstAudio === null) {
            firstAudio = (performance.now() - started) / 1000
            this.send({ type: 'audio_start', turn, response_id: rid, sample_rate: 24000 }, turn)
          }
          if (!phraseStarted) {
            phraseStarted = true
            response.spoken += phrase
            this.send({ type: 'assistant_delta', response_id: rid, text }, turn)
          }
          samples += chunk.length / 2
          this.send(chunk, turn)
        }
      }
      await producer
      if (ac.signal.aborted) throw ac.signal.reason
      if (llmError) throw llmError
      if (!spokenAny) throw new Error('回复没有可以说出来的内容')
      if (!samples) throw new Error('语音合成没有产生音频')
      const answer = cleanSpeech(response.spoken)
      response.recorded = true
      this.pushHistory('assistant', answer)
      this.send({ type: 'final_assistant', response_id: rid, text: answer, interrupted: false }, turn)
      if (this.deliveryId === rid) this.audioEnded = true
      this.send({
        type: 'audio_end', response_id: rid, samples, first_audio_seconds: firstAudio,
        generation_seconds: (performance.now() - started) / 1000,
      }, turn)
    } catch (error) {
      await producer
      if (turn !== this.turn || this.closed) {
        // Interrupted: interrupt()/close() already recorded what was actually spoken.
        const answer = cleanSpeech(response.spoken)
        if (!response.recorded && answer) this.pushHistory('assistant', answer)
        this.send({ type: 'final_assistant', response_id: rid, text: answer, interrupted: true })
        return
      }
      if (this.deliveryId === rid) this.deliveryId = null
      const detail = ac.signal.aborted && ac.signal.reason instanceof Error ? ac.signal.reason.message : error.message
      this.send({ type: 'error', detail, response_id: rid }, turn)
      ac.abort()
    } finally {
      clearTimeout(timer)
    }
  }

  /** Hang up: abort everything, save the call record. */
  close(reason) {
    if (this.closed) return
    this.send({ type: 'ended', reason })
    this.closed = true
    this.turn += 1
    this.abortResponse()
    this.asr?.close()
    if (this.ws.readyState === this.ws.OPEN) this.ws.close(1000)
    if (this.record.length) {
      try {
        const dir = join(this.dataDir, 'calls')
        mkdirSync(dir, { recursive: true })
        writeFileSync(join(dir, `${this.id}.json`), JSON.stringify({
          id: this.id, started: this.started, ended: new Date().toISOString(),
          persona: this.persona, topic: this.topic, messages: this.record,
        }, null, 2) + '\n')
      } catch {
        // a failed save must not keep the call open; nothing else to clean up
      }
    }
    this.resolveEnded()
  }
}
