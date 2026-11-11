// dsh-realtime-voice browser bundle: hand-written module-loader factory (no build step).
window.__ModuleLoader__.load({
  id: 'dsh-realtime-voice',
  factory(require) {
    const React = require('react')
    const { Button, Switch, Input, SegmentedControl, IconPlayOutlineRegular, IconTrashOutlineRegular } = require('@deepseek-ai/dsh-client-ui-primitives')
    const { createElement: h, useEffect, useRef, useState, useSyncExternalStore } = React

    const PANEL = 'realtime-voice.call'
    const API = '/api/realtime-voice'
    const DEFAULT_NAME = '大肥鱼'

    // ---------- tiny external store (shared by every component, survives panel switches) ----------

    function createStore(initial) {
      let value = initial
      const listeners = new Set()
      return {
        get: () => value,
        set(patch) { value = { ...value, ...patch }; listeners.forEach(fn => fn()) },
        subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn) },
      }
    }
    const useStore = store => useSyncExternalStore(store.subscribe, store.get, store.get)

    // App state = the host's State (SPEC §2); null until the first /state.
    const app = createStore({ state: null })
    const personaName = () => app.get().state?.settings?.persona?.name || DEFAULT_NAME

    async function api(path, body, init = {}) {
      const options = { credentials: 'same-origin', ...init }
      if (body !== undefined && !init.body) {
        options.method = 'POST'
        options.headers = { 'content-type': 'application/json' }
        options.body = JSON.stringify(body)
      }
      const res = await fetch(API + path, options)
      const type = res.headers.get('content-type') || ''
      if (!res.ok) {
        const detail = type.includes('json') ? (await res.json().catch(() => ({}))).detail : await res.text().catch(() => '')
        throw new Error(detail || `请求失败（HTTP ${res.status}）`)
      }
      return type.includes('json') ? res.json() : res.blob()
    }
    async function loadState() {
      const state = await api('/state')
      app.set({ state })
      return state
    }

    // One shared <audio> for voice samples and previews.
    let sampleAudio = null
    function playUrl(url) {
      if (sampleAudio) sampleAudio.pause()
      sampleAudio = new Audio(url)
      return sampleAudio.play()
    }

    // ---------- audio worklets (loaded from Blob URLs) ----------

    // Capture: resample whatever the context rate is to 16 kHz (linear), post 320-sample Int16 frames + RMS level.
    const CAPTURE_WORKLET = `
class RvCapture extends AudioWorkletProcessor {
  constructor() { super(); this.step = sampleRate / 16000; this.pos = 0; this.prev = 0; this.frame = new Int16Array(320); this.n = 0; this.sum = 0 }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (!ch || !ch.length) return true
    let pos = this.pos
    // x[-1] is the last sample of the previous block
    while (pos < ch.length - 1) {
      const i = Math.floor(pos), f = pos - i
      const a = i < 0 ? this.prev : ch[i]
      const s = Math.max(-1, Math.min(1, a + (ch[i + 1] - a) * f))
      this.frame[this.n++] = s < 0 ? s * 32768 : s * 32767
      this.sum += s * s
      if (this.n === 320) {
        this.port.postMessage({ pcm: this.frame.buffer, level: Math.sqrt(this.sum / 320) }, [this.frame.buffer])
        this.frame = new Int16Array(320); this.n = 0; this.sum = 0
      }
      pos += this.step
    }
    this.pos = pos - ch.length
    this.prev = ch[ch.length - 1]
    return true
  }
}
registerProcessor('rv-capture', RvCapture)
`
    // Playback: queue of Float32 chunks (24 kHz) and end markers; linear resample to the context rate.
    // A marker is reported ('done') once every sample queued before it has been read; 'flush' drops everything.
    const PLAYBACK_WORKLET = `
class RvPlayback extends AudioWorkletProcessor {
  constructor() {
    super(); this.q = []; this.off = 0; this.step = 24000 / sampleRate; this.frac = 0; this.cur = 0; this.next = 0; this.blocks = 0; this.sum = 0; this.count = 0
    this.port.onmessage = e => {
      const m = e.data
      if (m.type === 'pcm') this.q.push(m.samples)
      else if (m.type === 'mark') this.q.push(m.id)
      else if (m.type === 'flush') {
        const marks = this.q.filter(x => typeof x === 'string')
        this.q = []; this.off = 0; this.cur = 0; this.next = 0; this.frac = 0
        for (const id of marks) this.port.postMessage({ type: 'done', id })
      }
    }
  }
  pull() {
    while (this.q.length) {
      const head = this.q[0]
      if (typeof head === 'string') { this.q.shift(); this.port.postMessage({ type: 'done', id: head }); continue }
      if (this.off < head.length) return head[this.off++]
      this.q.shift(); this.off = 0
    }
    return 0
  }
  process(inputs, outputs) {
    const out = outputs[0][0]
    for (let i = 0; i < out.length; i++) {
      const s = this.cur + (this.next - this.cur) * this.frac
      out[i] = s; this.sum += s * s
      this.frac += this.step
      while (this.frac >= 1) { this.frac -= 1; this.cur = this.next; this.next = this.pull() }
    }
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out)
    this.count += out.length
    if (++this.blocks % 4 === 0) { this.port.postMessage({ type: 'level', level: Math.sqrt(this.sum / this.count) }); this.sum = 0; this.count = 0 }
    return true
  }
}
registerProcessor('rv-playback', RvPlayback)
`
    async function addWorklet(context, source) {
      const url = URL.createObjectURL(new Blob([source], { type: 'application/javascript' }))
      try { await context.audioWorklet.addModule(url) } finally { URL.revokeObjectURL(url) }
    }

    // ---------- call engine (module singleton, lives outside React) ----------

    const IDLE = { phase: 'idle', detail: '', error: '', muted: false, startedAt: 0, endedAt: 0, transcript: [], name: '', topic: '' }
    const call = createStore(IDLE)
    let session = null // { ws, stream, capture, playback, ... } while a call is up
    let entryId = 0
    // Live levels for the meter (written ~50/s, read by requestAnimationFrame, never through React).
    const levels = { mic: 0, out: 0 }

    function pushEntry(entry) {
      call.set({ transcript: [...call.get().transcript, { id: ++entryId, ...entry }].slice(-60) })
    }
    function updateEntry(match, patch) {
      const list = call.get().transcript
      for (let i = list.length - 1; i >= 0; i--) {
        if (!match(list[i])) continue
        const next = list.slice()
        next[i] = { ...list[i], ...(typeof patch === 'function' ? patch(list[i]) : patch) }
        call.set({ transcript: next })
        return true
      }
      return false
    }
    const dropPartial = () => call.set({ transcript: call.get().transcript.filter(e => !e.partial) })

    function micError(error) {
      if (!navigator.mediaDevices?.getUserMedia) return '当前页面不能使用麦克风：请通过 localhost 或 HTTPS 打开 DSH。'
      if (error?.name === 'NotAllowedError') return '麦克风权限被拒绝，请在浏览器里允许麦克风后重试。'
      if (error?.name === 'NotFoundError') return '没有找到麦克风。'
      return `无法打开麦克风：${error?.message || error}`
    }

    async function dial() {
      if (session) return
      const s = session = { closed: false, stopTurn: -1, dropAudio: false, ws: null, stream: null, capture: null, playback: null, playNode: null }
      call.set({ ...IDLE, phase: 'connecting', name: personaName(), transcript: [] })
      try {
        // Playback context first, inside the click gesture so it may start.
        s.playback = new AudioContext({ sampleRate: 24000 })
        await addWorklet(s.playback, PLAYBACK_WORKLET)
        s.playNode = new AudioWorkletNode(s.playback, 'rv-playback', { numberOfInputs: 0, outputChannelCount: [1] })
        s.playNode.connect(s.playback.destination)
        s.playNode.port.onmessage = e => onPlayback(s, e.data)

        const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false })
        // Hung up while the permission prompt was open: release the mic at once.
        if (s.closed) { stream.getTracks().forEach(t => t.stop()); return }
        s.stream = stream
        if (call.get().muted) stream.getAudioTracks().forEach(t => { t.enabled = false }) // muted before the mic arrived
        // Some browsers refuse a 16 kHz context or connecting a mic to it; the worklet then resamples.
        let source
        try {
          s.capture = new AudioContext({ sampleRate: 16000 })
          source = s.capture.createMediaStreamSource(s.stream)
        } catch {
          s.capture?.close().catch(() => {})
          s.capture = new AudioContext()
          source = s.capture.createMediaStreamSource(s.stream)
        }
        await addWorklet(s.capture, CAPTURE_WORKLET)
        if (s.closed) return
        const node = new AudioWorkletNode(s.capture, 'rv-capture', { numberOfOutputs: 0 })
        node.port.onmessage = e => {
          levels.mic = call.get().muted ? 0 : e.data.level
          if (s.ws?.readyState === WebSocket.OPEN && s.ready) s.ws.send(e.data.pcm)
        }
        source.connect(node)
        await Promise.all([s.capture.resume(), s.playback.resume()])
      } catch (error) {
        if (!s.closed) finish(s, { error: s.stream ? `音频初始化失败：${error?.message || error}` : micError(error) })
        return
      }
      openSocket(s)
    }

    function openSocket(s) {
      const ws = s.ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/realtime-voice/ws`)
      ws.binaryType = 'arraybuffer'
      ws.onopen = () => ws.send(JSON.stringify({ type: 'hello', client: 'dsh-realtime-voice' }))
      ws.onmessage = e => {
        if (s.closed) return
        if (typeof e.data !== 'string') return onAudio(s, e.data)
        let m
        try { m = JSON.parse(e.data) } catch { return }
        onMessage(s, m)
      }
      ws.onclose = e => {
        if (s.closed) return
        const reason = s.ready ? '通话连接已断开' : `无法建立通话连接${e.reason ? `：${e.reason}` : ''}`
        finish(s, { error: call.get().error || reason })
      }
    }

    function onMessage(s, m) {
      const state = call.get()
      switch (m.type) {
        case 'status':
          if (m.state !== 'ready') call.set({ phase: 'loading', detail: m.detail || '' })
          break
        case 'ready':
          s.ready = true
          call.set({ phase: 'listening', detail: '', startedAt: Date.now(), name: m.persona?.name || state.name, topic: m.topic || '' })
          break
        case 'speech_start':
          call.set({ userSpeaking: true })
          break
        case 'partial_user':
          if (!m.text) break
          if (!updateEntry(e => e.partial, { text: m.text })) pushEntry({ role: 'user', text: m.text, partial: true })
          break
        case 'final_user':
          dropPartial()
          pushEntry({ role: 'user', text: m.text })
          call.set({ phase: 'thinking', userSpeaking: false, error: '' })
          break
        case 'ignored_user':
          dropPartial()
          call.set({ userSpeaking: false })
          break
        case 'stop_audio':
          s.stopTurn = Math.max(s.stopTurn, m.turn ?? 0)
          s.dropAudio = true
          s.playNode?.port.postMessage({ type: 'flush' })
          updateEntry(e => e.role === 'assistant' && !e.done, { done: true, interrupted: true })
          if (state.phase === 'speaking') call.set({ phase: 'listening' })
          break
        case 'audio_start':
          // stop_audio carries the new turn; audio from an older turn is stale.
          s.dropAudio = typeof m.turn === 'number' && m.turn < s.stopTurn
          if (s.dropAudio) break
          pushEntry({ role: 'assistant', text: '', responseId: m.response_id })
          call.set({ phase: 'speaking', userSpeaking: false })
          break
        case 'assistant_delta':
          updateEntry(e => e.responseId === m.response_id, e => ({ text: e.text + m.text }))
          break
        case 'final_assistant':
          updateEntry(e => e.responseId === m.response_id, { text: m.text, done: true, interrupted: !!m.interrupted })
          break
        case 'audio_end':
          // playback_done goes out when the worklet has played everything queued before this marker.
          s.playNode?.port.postMessage({ type: 'mark', id: m.response_id })
          break
        case 'error':
          // A failed reply sends no final_assistant/audio_end: close its bubble and go back to listening.
          if (m.response_id) {
            updateEntry(e => e.responseId === m.response_id, { done: true })
            if (state.phase === 'thinking' || state.phase === 'speaking') call.set({ phase: 'listening' })
          }
          call.set({ error: m.detail || '出错了' })
          break
        case 'ended':
          finish(s, { info: m.reason ? `通话已结束：${m.reason}` : '' })
          break
      }
    }

    function onAudio(s, buffer) {
      if (s.dropAudio || !s.playNode || buffer.byteLength < 2) return
      const pcm = new Int16Array(buffer, 0, buffer.byteLength >> 1)
      const samples = new Float32Array(pcm.length)
      for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768
      s.playNode.port.postMessage({ type: 'pcm', samples }, [samples.buffer])
    }

    function onPlayback(s, m) {
      if (m.type === 'level') { levels.out = m.level; return }
      if (m.type !== 'done' || s.closed) return
      if (s.ws?.readyState === WebSocket.OPEN) s.ws.send(JSON.stringify({ type: 'playback_done', response_id: m.id }))
      if (call.get().phase === 'speaking') call.set({ phase: 'listening' })
    }

    function hangup() {
      if (!session) return
      if (session.ws?.readyState === WebSocket.OPEN) session.ws.send(JSON.stringify({ type: 'hangup' }))
      finish(session, {})
    }

    // Release everything this call holds; keeps the transcript on screen.
    function finish(s, { error = '', info = '' }) {
      if (s.closed) return
      s.closed = true
      if (session === s) session = null
      try { s.ws?.close(1000) } catch {}
      s.stream?.getTracks().forEach(t => t.stop())
      s.capture?.close().catch(() => {})
      s.playback?.close().catch(() => {})
      levels.mic = 0; levels.out = 0
      dropPartial()
      call.set({ phase: 'ended', detail: info, error, endedAt: Date.now(), userSpeaking: false })
      loadState().catch(() => {}) // pendingCall is cleared by the host
    }

    function setMuted(muted) {
      session?.stream?.getAudioTracks().forEach(t => { t.enabled = !muted })
      call.set({ muted })
    }

    // ---------- shared visuals ----------

    const svg = (size, ...children) => h('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }, ...children)
    const PHONE_PATH = 'M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z'
    const PhoneIcon = ({ size = 16 }) => svg(size, h('path', { d: PHONE_PATH }))
    const HangupIcon = ({ size = 16 }) => h('span', { style: { display: 'inline-flex', transform: 'rotate(135deg)' } }, svg(size, h('path', { d: PHONE_PATH })))
    const MicIcon = ({ size = 16 }) => svg(size, h('path', { d: 'M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z' }), h('path', { d: 'M19 10v2a7 7 0 0 1-14 0v-2' }), h('path', { d: 'M12 19v3' }))
    const MicOffIcon = ({ size = 16 }) => svg(size, h('path', { d: 'M2 2l20 20' }), h('path', { d: 'M18.89 13.23A7.12 7.12 0 0 0 19 12v-2' }), h('path', { d: 'M5 10v2a7 7 0 0 0 12 5' }), h('path', { d: 'M15 9.34V5a3 3 0 0 0-5.68-1.33' }), h('path', { d: 'M9 9v3a3 3 0 0 0 5.12 2.12' }), h('path', { d: 'M12 19v3' }))

    const CSS = `
.rv-round { border: 0; border-radius: 50%; corner-shape: round; display: inline-flex; align-items: center; justify-content: center; cursor: pointer; transition: transform .12s ease, filter .12s ease; }
.rv-round:focus-visible { outline: 2px solid var(--dsw-alias-link); outline-offset: 3px; }
.rv-round:active { transform: scale(.96); }
@media (hover: hover) { .rv-round:hover { filter: brightness(1.06); } }
.rv-dots::after { content: ''; animation: rv-dots 1.4s steps(4) infinite; }
@keyframes rv-dots { 0% { content: '' } 25% { content: '.' } 50% { content: '..' } 75% { content: '...' } }
@keyframes rv-breathe { 0%, 100% { transform: scale(1); opacity: .5 } 50% { transform: scale(1.06); opacity: .8 } }
@media (prefers-reduced-motion: reduce) { .rv-dots::after { animation: none; content: '…' } .rv-breathe { animation: none !important } }
`
    const styleTag = () => h('style', null, CSS)
    const muted = { color: 'var(--dsw-alias-label-tertiary)', fontSize: 13, lineHeight: '20px', margin: 0 }
    const formControl = { font: 'inherit', fontSize: 14, color: 'var(--dsw-alias-label-primary)', background: 'var(--dsw-alias-bg-base)', border: '0.5px solid var(--dsw-alias-border-l2)', borderRadius: 'var(--dsw-radius-md, 12px)', padding: '7px 10px', width: '100%', boxSizing: 'border-box' }

    function Avatar({ name, size = 112, active }) {
      return h('div', { style: { width: size, height: size, borderRadius: '50%', cornerShape: 'round', background: 'var(--dsw-alias-bg-layer-2)', boxShadow: active ? '0 0 0 0.5px var(--dsw-alias-border-l2)' : 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: size * 0.4, fontWeight: 500, color: 'var(--dsw-alias-label-primary)', position: 'relative', zIndex: 1 } }, Array.from(name || DEFAULT_NAME)[0])
    }

    // ---------- call page ----------

    const PHASE_TEXT = { idle: '', connecting: '连接中', loading: '正在加载模型…', listening: '正在听', thinking: '思考中', speaking: '正在说话', ended: '通话已结束' }
    const clock = ms => { const t = Math.max(0, Math.floor(ms / 1000)); return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}` }

    // Pulse ring around the avatar driven by the live mic / output level.
    function LevelRing({ phase }) {
      const ref = useRef(null)
      useEffect(() => {
        let frame = 0, smooth = 0
        const tick = () => {
          const raw = phase === 'speaking' ? levels.out : phase === 'listening' || phase === 'thinking' ? levels.mic : 0
          smooth += (Math.min(1, raw * 6) - smooth) * 0.25
          if (ref.current) {
            ref.current.style.transform = `scale(${1 + smooth * 0.35})`
            ref.current.style.opacity = String(0.15 + smooth * 0.6)
          }
          frame = requestAnimationFrame(tick)
        }
        frame = requestAnimationFrame(tick)
        return () => cancelAnimationFrame(frame)
      }, [phase])
      const color = phase === 'speaking' ? 'var(--dsw-alias-state-success-primary)' : 'var(--dsw-alias-link)'
      return h('div', { ref, style: { position: 'absolute', inset: -10, borderRadius: '50%', cornerShape: 'round', border: `2px solid ${color}`, opacity: 0.15, pointerEvents: 'none' } })
    }

    function Transcript({ entries }) {
      const ref = useRef(null)
      useEffect(() => { const el = ref.current; if (el) el.scrollTop = el.scrollHeight }, [entries])
      if (!entries.length) return h('div', { style: { flex: 1 } })
      return h('div', { ref, 'aria-live': 'polite', style: { flex: 1, minHeight: 0, overflowY: 'auto', width: '100%', display: 'flex', flexDirection: 'column', gap: 10, padding: '4px 2px' } },
        entries.map(e => h('div', { key: e.id, style: { alignSelf: e.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '82%' } },
          h('div', { style: {
            fontSize: 14, lineHeight: '22px', padding: '8px 12px', borderRadius: 'var(--dsw-radius-lg, 16px)',
            background: e.role === 'user' ? 'var(--dsw-alias-bg-layer-2)' : 'transparent',
            color: e.partial ? 'var(--dsw-alias-label-tertiary)' : 'var(--dsw-alias-label-primary)',
            fontStyle: e.partial ? 'italic' : 'normal', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          } }, e.text || '…'),
          e.interrupted && h('div', { style: { ...muted, fontSize: 12, padding: '0 12px' } }, '已打断'))))
    }

    function RoundButton({ label, size, fill, color, onClick, children, pressed }) {
      return h('div', { style: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 } },
        h('button', { type: 'button', className: 'rv-round', 'aria-label': label, 'aria-pressed': pressed, onClick, style: { width: size, height: size, background: fill, color } }, children),
        h('span', { style: { ...muted, fontSize: 12 } }, label))
    }

    function CallPage() {
      const c = useStore(call)
      const { state } = useStore(app)
      const [, tick] = useState(0)
      const live = !['idle', 'ended'].includes(c.phase)
      useEffect(() => { if (!state) loadState().catch(() => {}) }, [])
      useEffect(() => {
        if (!c.startedAt || !live) return
        const id = setInterval(() => tick(n => n + 1), 1000)
        return () => clearInterval(id)
      }, [c.startedAt, live])
      const name = live || c.phase === 'ended' ? c.name || personaName() : personaName()
      const topic = live ? c.topic : state?.pendingCall?.topic || ''
      let status = PHASE_TEXT[c.phase]
      if (c.phase === 'listening' && c.userSpeaking) status = '正在听…'
      if (c.phase === 'ended' && c.startedAt) status += ` · ${clock(c.endedAt - c.startedAt)}`
      const showTimer = live && c.startedAt
      const waiting = c.phase === 'connecting' || c.phase === 'loading' || c.phase === 'thinking'

      return h('div', { style: { height: '100%', boxSizing: 'border-box', background: 'var(--dsw-alias-bg-base)', color: 'var(--dsw-alias-label-primary)', display: 'flex', justifyContent: 'center', padding: 'calc(var(--dsh-frame-top-clearance, 0px) + 32px) 16px 32px', overflow: 'hidden' } },
        styleTag(),
        h('div', { style: { width: '100%', maxWidth: 440, height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 24 } },
          h('div', { style: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, paddingTop: 16 } },
            h('div', { style: { position: 'relative' } },
              live && h(LevelRing, { phase: c.phase }),
              live && waiting && h('div', { className: 'rv-breathe', style: { position: 'absolute', inset: -10, borderRadius: '50%', cornerShape: 'round', background: 'var(--dsw-alias-bg-layer-2)', animation: 'rv-breathe 2.4s ease-in-out infinite' } }),
              h(Avatar, { name, active: live })),
            h('div', { style: { fontSize: 24, lineHeight: '32px', fontWeight: 600 } }, name),
            h('div', { role: 'status', style: { fontSize: 14, lineHeight: '20px', color: 'var(--dsw-alias-label-secondary)', minHeight: 20, textAlign: 'center' } },
              showTimer && h('span', { style: { fontVariantNumeric: 'tabular-nums' } }, clock(Date.now() - c.startedAt), status ? ' · ' : ''),
              status && h('span', { className: waiting ? 'rv-dots' : undefined }, status.replace(/…$/, waiting ? '' : '…'))),
            c.phase === 'loading' && c.detail && h('p', { style: { ...muted, textAlign: 'center' } }, c.detail),
            c.phase === 'ended' && c.detail && h('p', { style: { ...muted, textAlign: 'center' } }, c.detail),
            topic && h('p', { style: { ...muted, textAlign: 'center' } }, `话题：${topic}`),
            c.error && h('p', { role: 'alert', style: { margin: 0, fontSize: 13, lineHeight: '20px', color: 'var(--dsw-alias-state-error-primary)', textAlign: 'center', maxWidth: 360 } }, c.error)),
          h(Transcript, { entries: c.transcript }),
          live
            ? h('div', { style: { display: 'flex', gap: 48, alignItems: 'flex-start' } },
              h(RoundButton, { label: c.muted ? '取消静音' : '静音', size: 60, pressed: c.muted, fill: c.muted ? 'var(--dsw-alias-label-primary)' : 'var(--dsw-alias-bg-layer-2)', color: c.muted ? 'var(--dsw-alias-bg-base)' : 'var(--dsw-alias-label-primary)', onClick: () => setMuted(!c.muted) }, h(c.muted ? MicOffIcon : MicIcon, { size: 24 })),
              h(RoundButton, { label: '挂断', size: 60, fill: 'var(--dsw-alias-state-error-primary)', color: 'var(--dsw-alias-label-primary-foreground)', onClick: hangup }, h(HangupIcon, { size: 26 })))
            : h(RoundButton, { label: c.phase === 'ended' ? '重新拨打' : '拨打', size: 72, fill: 'var(--dsw-alias-state-success-primary)', color: 'var(--dsw-alias-label-primary-foreground)', onClick: () => { void dial() } }, h(PhoneIcon, { size: 28 }))))
    }

    // ---------- settings page ----------

    // 'a.b.c' + value → { a: { b: { c: value } } }
    const patchAt = (path, value) => path.split('.').reduceRight((acc, key) => ({ [key]: acc }), value)
    const getAt = (obj, path) => path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj)

    // plain: a wrapper that is not a <label> (when the control is a button).
    function Field({ label, hint, children, plain }) {
      return h(plain ? 'div' : 'label', { style: { display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0 } },
        h('span', { style: { fontSize: 13, lineHeight: '20px', color: 'var(--dsw-alias-label-secondary)' } }, label),
        children,
        hint && h('span', { style: muted }, hint))
    }
    const grid = { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }

    // Text/number field that saves on blur when the value changed.
    function TextField({ label, hint, value, onSave, number, multiline, placeholder, type }) {
      const shown = value == null ? '' : String(value)
      const [text, setText] = useState(shown)
      useEffect(() => setText(shown), [shown])
      const commit = () => {
        if (text === shown) return
        if (number) {
          const n = Number(text)
          if (text.trim() === '' || !Number.isFinite(n)) return setText(shown)
          onSave(n)
        } else onSave(text)
      }
      const props = { value: text, placeholder, type: type || (number ? 'number' : 'text'), step: number ? 'any' : undefined, onChange: e => setText(e.target.value), onBlur: commit }
      return h(Field, { label, hint }, multiline
        ? h('textarea', { ...props, rows: multiline, style: { ...formControl, resize: 'vertical', lineHeight: '22px' } })
        : h(Input, { ...props, onKeyDown: e => { if (e.key === 'Enter') e.currentTarget.blur() } }))
    }

    function SelectField({ label, hint, value, options, onChange, disabled }) {
      const list = options.some(o => o.value === value) ? options : [{ value, label: value || '（未选择）' }, ...options]
      return h(Field, { label, hint }, h('select', { value, disabled, onChange: e => onChange(e.target.value), style: formControl },
        list.map(o => h('option', { key: o.value, value: o.value }, o.label))))
    }

    function SecretField({ label, name, saved, run }) {
      const [text, setText] = useState('')
      const save = value => run(`保存${label}`, () => api('/secrets', { name, value }).then(state => { app.set({ state }); setText('') }))
      return h(Field, { plain: saved, label, hint: saved ? '已保存（不会显示）' : '未设置' },
        h('div', { style: { display: 'flex', gap: 8 } },
          h(Input, { type: 'password', autoComplete: 'off', value: text, placeholder: saved ? '••••••••（输入新值替换）' : '粘贴密钥', onChange: e => setText(e.target.value), onBlur: () => { if (text) save(text) }, onKeyDown: e => { if (e.key === 'Enter') e.currentTarget.blur() } }),
          saved && h(Button, { variant: 'outline', onClick: () => save('') }, '清除')))
    }

    function Section({ title, children, aside }) {
      return h('section', { style: { border: '0.5px solid var(--dsw-alias-settings-card-stroke)', background: 'var(--dsw-alias-settings-card-fill)', borderRadius: 'var(--dsw-radius-xl, 20px)', padding: 20, display: 'flex', flexDirection: 'column', gap: 14 } },
        h('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 } },
          h('h3', { style: { margin: 0, fontSize: 16, lineHeight: '24px', fontWeight: 600 } }, title), aside),
        children)
    }

    const LANGS = [{ value: 'zh', label: '中文' }, { value: 'en', label: '英文' }]
    const ENGINE_TEXT = { stopped: '已停止', starting: '启动中', warming: '加载模型中', ready: '运行中', error: '出错' }
    const ENGINE_COLOR = { stopped: 'var(--dsw-alias-state-idle-primary)', starting: 'var(--dsw-alias-state-warn-primary)', warming: 'var(--dsw-alias-state-warn-primary)', ready: 'var(--dsw-alias-state-success-primary)', error: 'var(--dsw-alias-state-error-primary)' }
    const providerHealth = p => !p ? '' : `${p.provider} · ${p.error ? `错误：${p.error}` : p.ready ? '已就绪' : p.loading ? '加载中' : '未加载'}`

    function SettingsPage() {
      const { state } = useStore(app)
      const [notice, setNotice] = useState(null) // { text, error }
      const [busy, setBusy] = useState('')
      const [edgeVoices, setEdgeVoices] = useState(null)
      const [voices, setVoices] = useState(null)
      const [calls, setCalls] = useState(null)
      const [openCall, setOpenCall] = useState(null)
      const [preview, setPreview] = useState('你好，我是大肥鱼。今天想聊点什么？')
      const [upload, setUpload] = useState({ name: '', language: 'zh', text: '' })
      const [design, setDesign] = useState({ name: '', language: 'zh', instruction: '' })
      const fileRef = useRef(null)
      const engineState = state?.engine?.state

      // Run one action with a busy label and a visible result.
      async function run(label, fn, done) {
        setBusy(label); setNotice(null)
        try {
          const value = await fn()
          if (done) setNotice({ text: done })
          return value
        } catch (error) {
          setNotice({ text: `${label}失败：${error?.message || error}`, error: true })
        } finally { setBusy('') }
      }
      const save = (path, value) => run('保存设置', () => api('/settings', patchAt(path, value)).then(next => app.set({ state: next })))
      const loadVoices = () => run('读取音色库', () => api('/voices').then(r => setVoices(r.voices || [])))
      const loadEdge = () => run('读取 Edge 声音列表', () => api('/tts/voices?provider=edge').then(r => setEdgeVoices(r.voices || [])))

      useEffect(() => {
        loadState().catch(error => setNotice({ text: `读取设置失败：${error.message}`, error: true }))
        api('/calls').then(r => setCalls(r.calls || [])).catch(() => setCalls([]))
      }, [])
      // Poll while the engine is starting or warming.
      useEffect(() => {
        if (engineState !== 'starting' && engineState !== 'warming') return
        const id = setInterval(() => loadState().catch(() => {}), 2000)
        return () => clearInterval(id)
      }, [engineState])
      // Engine-backed lists load by themselves once the engine runs (fetching them otherwise starts it).
      useEffect(() => {
        if (engineState !== 'ready') return
        if (voices === null) api('/voices').then(r => setVoices(r.voices || [])).catch(() => {})
        if (edgeVoices === null && state?.settings?.tts?.provider === 'edge') api('/tts/voices?provider=edge').then(r => setEdgeVoices(r.voices || [])).catch(() => {})
      }, [engineState, state?.settings?.tts?.provider])

      if (!state) return h('div', { style: { padding: 24 } }, styleTag(), notice ? h('p', { role: 'alert', style: { color: 'var(--dsw-alias-state-error-primary)' } }, notice.text) : h('p', { style: muted }, '正在读取设置…'))

      const st = state.settings
      const text = (label, path, extra = {}) => h(TextField, { key: path, label, value: getAt(st, path), onSave: v => save(path, v), ...extra })
      const num = (label, path, extra = {}) => text(label, path, { number: true, ...extra })
      const select = (label, path, options, extra = {}) => h(SelectField, { key: path, label, value: getAt(st, path) ?? '', options, onChange: v => save(path, v), ...extra })
      const segmented = (id, label, path, options) => h(SegmentedControl, { id: `rv-${id}`, label, value: getAt(st, path), options, onChange: v => save(path, v), disabled: !!busy })
      const secret = (label, name) => h(SecretField, { label, name, saved: !!state.secrets?.[name], run })

      const catalog = state.llmCatalog || []
      const llmProvider = catalog.find(p => p.id === st.llm.dsh.provider)
      const engine = state.engine || {}
      const breezeVoices = (voices || []).map(v => ({ value: v.id, label: `${v.name}（${v.language === 'zh' ? '中文' : '英文'}）` }))
      const edgeOptions = (edgeVoices || []).map(v => ({ value: v.id, label: `${v.name || v.id} · ${v.language}${v.gender ? ` · ${v.gender === 'Female' ? '女' : v.gender === 'Male' ? '男' : v.gender}` : ''}` }))

      async function playPreview() {
        await run('试听', async () => {
          const blob = await api('/tts/preview', { text: preview })
          await playUrl(URL.createObjectURL(blob))
        })
      }
      async function submitUpload() {
        const file = fileRef.current?.files?.[0]
        if (!file) return setNotice({ text: '请先选择一段音频文件（3–20 秒）。', error: true })
        if (!upload.name.trim()) return setNotice({ text: '请填写音色名称。', error: true })
        const query = new URLSearchParams({ name: upload.name.trim(), language: upload.language, text: upload.text.trim() })
        await run('上传音色', async () => {
          await api(`/voices/upload?${query}`, undefined, { method: 'POST', headers: { 'content-type': file.type || 'application/octet-stream' }, body: file })
          setUpload({ name: '', language: upload.language, text: '' })
          if (fileRef.current) fileRef.current.value = ''
          setVoices((await api('/voices')).voices || [])
        }, '音色已上传。')
      }
      async function submitDesign() {
        if (!design.name.trim() || !design.instruction.trim()) return setNotice({ text: '请填写音色名称和描述。', error: true })
        await run('设计音色', async () => {
          await api('/voices/design', { name: design.name.trim(), language: design.language, instruction: design.instruction.trim(), text: null, seed: null })
          setDesign({ name: '', language: design.language, instruction: '' })
          setVoices((await api('/voices')).voices || [])
        }, '音色已生成。')
      }
      async function removeVoice(v) {
        if (!window.confirm(`删除音色「${v.name}」？此操作不能撤销。`)) return
        await run('删除音色', async () => {
          await api('/voices/delete', { id: v.id })
          setVoices((await api('/voices')).voices || [])
        })
      }
      async function toggleCall(id) {
        if (openCall?.id === id) return setOpenCall(null)
        await run('读取通话记录', async () => setOpenCall(await api(`/calls/get?id=${encodeURIComponent(id)}`)))
      }

      return h('div', { style: { display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 760, paddingBottom: 32, color: 'var(--dsw-alias-label-primary)' } },
        styleTag(),
        h('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 } },
          h('div', null,
            h('h2', { style: { margin: 0, fontSize: 20, lineHeight: '28px', fontWeight: 600 } }, '实时语音'),
            h('p', { style: muted }, `和${st.persona.name || DEFAULT_NAME}打电话：浏览器采集声音，本机引擎识别与合成，模型实时回复。`)),
          h(Button, { variant: 'primary', icon: h(PhoneIcon, { size: 16 }), onClick: () => openCallPage() }, '打开通话')),
        (notice || busy) && h('div', { role: notice?.error ? 'alert' : 'status', style: { position: 'sticky', top: 0, zIndex: 2, padding: '8px 12px', borderRadius: 'var(--dsw-radius-md, 12px)', background: 'var(--dsw-alias-bg-layer-1)', boxShadow: 'var(--dsw-elevation-panel)', fontSize: 13, lineHeight: '20px', color: notice?.error ? 'var(--dsw-alias-state-error-primary)' : 'var(--dsw-alias-label-secondary)' } },
          busy ? h('span', { className: 'rv-dots' }, `${busy}中`) : notice.text),

        h(Section, { title: '角色' },
          h('div', { style: grid }, text('名字', 'persona.name'), text('开场白', 'persona.greeting', { hint: '接通后先说的话，留空则等你先开口' })),
          text('人设提示词', 'persona.prompt', { multiline: 6 })),

        h(Section, { title: '语言模型' },
          segmented('llm', '语言模型来源', 'llm.provider', [{ value: 'dsh', label: 'DSH 已配置的模型' }, { value: 'openai', label: 'OpenAI 兼容接口' }]),
          st.llm.provider === 'dsh'
            ? h('div', { style: grid },
              select('服务商', 'llm.dsh.provider', [{ value: '', label: '默认（第一个可用）' }, ...catalog.map(p => ({ value: p.id, label: p.name || p.id }))]),
              select('模型', 'llm.dsh.model', [{ value: '', label: '默认' }, ...(llmProvider?.models || []).map(m => ({ value: m.id, label: m.name || m.id }))]))
            : h('div', { style: grid }, text('接口地址', 'llm.openai.baseUrl'), text('模型', 'llm.openai.model'), secret('API Key', 'llmOpenaiKey')),
          catalog.length === 0 && st.llm.provider === 'dsh' && h('p', { style: muted }, '没有读到 DSH 的模型列表，将使用默认模型。'),
          h('div', { style: grid }, num('最多输出 Token', 'llm.maxTokens'), num('温度', 'llm.temperature'))),

        h(Section, { title: '语音识别' },
          segmented('asr', '语音识别来源', 'asr.provider', [{ value: 'qwen3', label: 'Qwen3-ASR（本机 GPU）' }, { value: 'openai', label: '云端（OpenAI 兼容）' }]),
          st.asr.provider === 'qwen3'
            ? h('div', { style: grid }, text('Qwen3 模型', 'asr.qwen3.model', { hint: '本地目录或 Hugging Face 名称' }), text('设备', 'asr.qwen3.device'))
            : h('div', { style: grid }, text('接口地址', 'asr.openai.baseUrl'), text('模型', 'asr.openai.model'), secret('API Key', 'asrOpenaiKey')),
          h('div', { style: grid },
            select('识别语言', 'asr.language', [{ value: 'auto', label: '自动' }, ...LANGS]),
            num('实时字幕间隔（秒）', 'asr.partialInterval')),
          h('div', { style: { display: 'flex', alignItems: 'center', gap: 10 } },
            h(Switch, { checked: !!st.asr.partials, label: '实时字幕', onChange: v => save('asr.partials', v), disabled: !!busy }),
            h('span', { style: { fontSize: 14 } }, '说话时显示实时字幕（也用于判断打断）')),
          h('details', null,
            h('summary', { style: { cursor: 'pointer', fontSize: 13, color: 'var(--dsw-alias-label-secondary)' } }, '断句与打断参数'),
            h('div', { style: { ...grid, marginTop: 12 } },
              num('静音判定（秒）', 'turn.silence'), num('确认时长（秒）', 'turn.confirm'),
              num('问句静音（秒）', 'turn.questionSilence'), num('问句确认（秒）', 'turn.questionConfirm'),
              num('最短语音（秒）', 'turn.minSpeech'), num('VAD 阈值', 'turn.vadThreshold'),
              num('前置缓冲（秒）', 'turn.preRoll'), num('提前识别（秒）', 'turn.earlyTranscribe'),
              num('打断所需字数', 'bargeIn.minWords', { hint: '0 = 一开口就打断' })))),

        h(Section, { title: '语音合成' },
          segmented('tts', '语音合成来源', 'tts.provider', [{ value: 'edge', label: 'Edge 在线' }, { value: 'breeze', label: 'Breeze 克隆（本机 GPU）' }, { value: 'openai', label: 'OpenAI 兼容' }]),
          st.tts.provider === 'edge' && h('div', { style: grid },
            edgeVoices
              ? select('声音', 'tts.edge.voice', edgeOptions)
              : h(Field, { plain: true, label: '声音' }, h('div', { style: { display: 'flex', gap: 8, alignItems: 'center' } },
                h('span', { style: { fontSize: 14, flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' } }, st.tts.edge.voice),
                h(Button, { variant: 'outline', size: 'sm', disabled: !!busy, onClick: () => { void loadEdge() } }, '加载列表'))),
            text('语速', 'tts.edge.rate', { hint: '如 +10%、-5%' }), text('音调', 'tts.edge.pitch', { hint: '如 +0Hz、-20Hz' })),
          st.tts.provider === 'openai' && h('div', { style: grid },
            text('接口地址', 'tts.openai.baseUrl'), text('模型', 'tts.openai.model'), text('声音', 'tts.openai.voice'), secret('API Key', 'ttsOpenaiKey')),
          st.tts.provider === 'breeze' && h('div', { style: { display: 'flex', flexDirection: 'column', gap: 12 } },
            h('div', { style: grid },
              voices
                ? select('音色', 'tts.breeze.voiceId', breezeVoices)
                : h(Field, { plain: true, label: '音色' }, h(Button, { variant: 'outline', size: 'sm', disabled: !!busy, onClick: () => { void loadVoices() } }, '加载音色库')),
              num('Worker 端口', 'tts.breeze.port')),
            text('Breeze Python', 'tts.breeze.python', { hint: 'Breeze 自己的虚拟环境里的 python' }),
            text('Breeze 代码目录', 'tts.breeze.codeDir'),
            text('Breeze 模型目录', 'tts.breeze.model')),
          h('div', { style: { display: 'flex', gap: 8, alignItems: 'flex-end' } },
            h('div', { style: { flex: 1, minWidth: 0 } }, h(Field, { label: '试听文本' }, h(Input, { value: preview, onChange: e => setPreview(e.target.value) }))),
            h(Button, { variant: 'outline', icon: h(IconPlayOutlineRegular, { size: 16 }), disabled: !!busy || !preview.trim(), onClick: () => { void playPreview() } }, '试听'))),

        h(Section, { title: '音色库', aside: h(Button, { variant: 'ghost', size: 'sm', disabled: !!busy, onClick: () => { void loadVoices() } }, voices ? '刷新' : '加载') },
          h('p', { style: muted }, 'Breeze 克隆用的参考声音。上传一段 3–20 秒的清晰人声，或用一句描述生成新音色。'),
          voices === null
            ? h('p', { style: muted }, engineState === 'ready' ? '正在读取…' : '引擎未运行；点「加载」会启动引擎。')
            : voices.length === 0
              ? h('p', { style: muted }, '还没有音色。')
              : h('div', { style: { display: 'flex', flexDirection: 'column' } }, voices.map((v, i) => h('div', { key: v.id, style: { display: 'flex', alignItems: 'center', gap: 12, padding: '10px 0', borderTop: i ? '0.5px solid var(--dsw-alias-border-l2)' : 'none' } },
                h(Avatar, { name: v.name, size: 32 }),
                h('div', { style: { flex: 1, minWidth: 0 } },
                  h('div', { style: { fontSize: 14, lineHeight: '20px', fontWeight: 500 } }, v.name, v.id === st.tts.breeze.voiceId && h('span', { style: { ...muted, marginLeft: 8 } }, '使用中')),
                  h('div', { style: { ...muted, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } },
                    `${v.language === 'zh' ? '中文' : '英文'} · ${v.source === 'design' ? '设计' : '上传'} · ${Number(v.duration || 0).toFixed(1)} 秒${v.text ? ` · ${v.text}` : ''}`)),
                v.id !== st.tts.breeze.voiceId && h(Button, { variant: 'ghost', size: 'sm', disabled: !!busy, onClick: () => save('tts.breeze.voiceId', v.id) }, '使用'),
                h(Button, { variant: 'ghost', size: 'sm', 'aria-label': `播放 ${v.name}`, icon: h(IconPlayOutlineRegular, { size: 16 }), onClick: () => { playUrl(`${API}/voices/audio?id=${encodeURIComponent(v.id)}`).catch(error => setNotice({ text: `播放失败：${error.message}`, error: true })) } }),
                h(Button, { variant: 'ghost', size: 'sm', 'aria-label': `删除 ${v.name}`, icon: h(IconTrashOutlineRegular, { size: 16 }), disabled: !!busy, onClick: () => { void removeVoice(v) } })))),
          h('details', null,
            h('summary', { style: { cursor: 'pointer', fontSize: 14 } }, '上传声音片段'),
            h('div', { style: { display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 } },
              h(Field, { label: '音频文件' }, h('input', { ref: fileRef, type: 'file', accept: 'audio/*,video/*', style: { fontSize: 14 } })),
              h('div', { style: grid },
                h(Field, { label: '名称' }, h(Input, { value: upload.name, onChange: e => setUpload({ ...upload, name: e.target.value }) })),
                h(Field, { label: '语言' }, h('select', { value: upload.language, onChange: e => setUpload({ ...upload, language: e.target.value }), style: formControl }, LANGS.map(o => h('option', { key: o.value, value: o.value }, o.label))))),
              h(Field, { label: '片段文字（可选）', hint: '留空会自动识别' }, h('textarea', { rows: 2, value: upload.text, onChange: e => setUpload({ ...upload, text: e.target.value }), style: { ...formControl, resize: 'vertical' } })),
              h('div', null, h(Button, { variant: 'primary', disabled: !!busy, onClick: () => { void submitUpload() } }, '上传')))),
          h('details', null,
            h('summary', { style: { cursor: 'pointer', fontSize: 14 } }, '用描述设计音色'),
            h('div', { style: { display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 } },
              h('div', { style: grid },
                h(Field, { label: '名称' }, h(Input, { value: design.name, onChange: e => setDesign({ ...design, name: e.target.value }) })),
                h(Field, { label: '语言' }, h('select', { value: design.language, onChange: e => setDesign({ ...design, language: e.target.value }), style: formControl }, LANGS.map(o => h('option', { key: o.value, value: o.value }, o.label))))),
              h(Field, { label: '声音描述', hint: '需要已配置 Breeze；生成可能要一两分钟' }, h('textarea', { rows: 3, value: design.instruction, placeholder: '例如：低沉慵懒的中年男声，语速偏慢，带点幽默', onChange: e => setDesign({ ...design, instruction: e.target.value }), style: { ...formControl, resize: 'vertical' } })),
              h('div', null, h(Button, { variant: 'primary', disabled: !!busy, onClick: () => { void submitDesign() } }, '生成'))))),

        h(Section, { title: '引擎', aside: h('div', { style: { display: 'flex', gap: 8 } },
          h(Button, { variant: 'outline', size: 'sm', disabled: !!busy || engineState === 'starting' || engineState === 'warming', onClick: () => { void run('启动引擎', () => api('/engine/start', {}).then(next => app.set({ state: next }))) } }, engineState === 'ready' ? '重启' : '启动'),
          h(Button, { variant: 'ghost', size: 'sm', disabled: !!busy || engineState === 'stopped', onClick: () => { void run('停止引擎', () => api('/engine/stop', {}).then(next => app.set({ state: next }))) } }, '停止')) },
          h('div', { style: { display: 'flex', alignItems: 'center', gap: 8, fontSize: 14 } },
            h('span', { style: { width: 8, height: 8, borderRadius: '50%', cornerShape: 'round', background: ENGINE_COLOR[engineState] || ENGINE_COLOR.stopped } }),
            ENGINE_TEXT[engineState] || engineState || '未知',
            engine.health && h('span', { style: muted }, `ASR ${providerHealth(engine.health.asr)} · TTS ${providerHealth(engine.health.tts)}`)),
          engine.error && h('p', { role: 'alert', style: { margin: 0, fontSize: 13, color: 'var(--dsw-alias-state-error-primary)' } }, engine.error),
          h('div', { style: grid }, text('Python', 'engine.python', { hint: '留空：仓库 .venv 或 python3' }), num('端口', 'engine.port'), num('空闲自动停止（分钟）', 'engine.idleMinutes', { hint: '0 = 不自动停止' })),
          h('details', null,
            h('summary', { style: { cursor: 'pointer', fontSize: 13, color: 'var(--dsw-alias-label-secondary)' } }, '引擎日志'),
            h('pre', { style: { margin: '10px 0 0', maxHeight: 280, overflow: 'auto', fontSize: 12, lineHeight: '18px', padding: 12, borderRadius: 'var(--dsw-radius-lg, 16px)', background: 'var(--dsw-alias-bg-base)', color: 'var(--dsw-alias-label-secondary)', whiteSpace: 'pre-wrap', wordBreak: 'break-all' } }, (engine.log || []).join('\n') || '（暂无日志）')),
          h('p', { style: muted }, `数据目录：${state.dataDir || ''}`)),

        h(Section, { title: '通话记录', aside: h(Button, { variant: 'ghost', size: 'sm', onClick: () => { api('/calls').then(r => setCalls(r.calls || [])).catch(() => {}) } }, '刷新') },
          !calls ? h('p', { style: muted }, '正在读取…') : calls.length === 0 ? h('p', { style: muted }, '还没有通话。')
            : h('div', { style: { display: 'flex', flexDirection: 'column' } }, calls.map((item, i) => h('div', { key: item.id, style: { borderTop: i ? '0.5px solid var(--dsw-alias-border-l2)' : 'none', padding: '10px 0' } },
              h('button', { type: 'button', onClick: () => { void toggleCall(item.id) }, style: { all: 'unset', cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: 2, width: '100%' } },
                h('span', { style: { fontSize: 14, lineHeight: '20px' } }, `${new Date(item.started).toLocaleString()} · ${item.persona || ''}${item.topic ? ` · ${item.topic}` : ''}`),
                h('span', { style: { ...muted, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } }, `${item.turns ?? 0} 轮${item.ended ? ` · ${clock(Date.parse(item.ended) - Date.parse(item.started))}` : ''}${item.preview ? ` · ${item.preview}` : ''}`)),
              openCall?.id === item.id && h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8, paddingLeft: 12, borderLeft: '1px solid var(--dsw-alias-border-l2)' } },
                (openCall.messages || []).map((m, j) => h('p', { key: j, style: { margin: 0, fontSize: 13, lineHeight: '20px' } },
                  h('span', { style: { color: 'var(--dsw-alias-label-tertiary)' } }, m.role === 'user' ? '我：' : `${openCall.persona || '对方'}：`), m.text))))))))
    }

    // ---------- launch points ----------

    let openCallPage = () => {}

    function HeroOption() {
      useStore(app)
      return h(Button, { variant: 'outline', size: 'sm', onClick: () => openCallPage() }, `📞 打电话给${personaName()}`)
    }

    // Tool card: auto-opens the call page once for a fresh result.
    const autoOpened = new Set()
    function ToolCard({ phase, block, callId }) {
      useStore(app)
      const fresh = phase === 'result' && !block?.isError && Date.now() - (block?.time || 0) < 30000
      useEffect(() => {
        if (!fresh || autoOpened.has(callId)) return
        autoOpened.add(callId)
        loadState().catch(() => {})
        openCallPage()
      }, [fresh, callId])
      const topic = block?.args?.text?.('topic') || ''
      const error = phase === 'result' && block?.isError ? (block.content || []).map(c => c.text || '').join('').trim() || '打开通话失败' : ''
      const name = personaName()
      return h('div', { style: { display: 'flex', alignItems: 'center', gap: 12, padding: 12, maxWidth: 420, borderRadius: 'var(--dsw-radius-lg, 16px)', border: '0.5px solid var(--dsw-alias-border-l2)', background: 'var(--dsw-alias-bg-layer-1)', color: 'var(--dsw-alias-label-primary)' } },
        h(Avatar, { name, size: 40 }),
        h('div', { style: { flex: 1, minWidth: 0 } },
          h('div', { style: { fontSize: 14, lineHeight: '20px', fontWeight: 500 } }, `打电话给${name}`),
          h('div', { style: { ...muted, color: error ? 'var(--dsw-alias-state-error-primary)' : muted.color, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } },
            error || (phase === 'result' ? (topic ? `话题：${topic}` : '通话页面已打开') : '正在准备通话…'))),
        h(Button, { variant: 'primary', size: 'sm', icon: h(PhoneIcon, { size: 14 }), disabled: phase !== 'result' || !!error, onClick: () => { openCallPage(); void dial() } }, '接通'))
    }

    return {
      name: 'dsh-realtime-voice-client',
      inject: ['slots', 'layout'],
      apply(ctx) {
        openCallPage = () => ctx.layout.selectPanel(PANEL)
        let alive = true
        ctx.effect(() => () => { alive = false })

        ctx.slots.inject('settings.section', () => ctx.slots.register({ name: 'settings.section', id: 'realtime-voice', order: 17, label: () => '实时语音' }, SettingsPage))
        ctx.slots.inject('main', () => ctx.slots.register({ name: 'main', key: PANEL }, CallPage))
        ctx.slots.inject('conversation.hero.options', () => ctx.slots.register({ name: 'conversation.hero.options', id: 'realtime-voice.call' }, HeroOption))
        ctx.slots.inject('tool.call.toolview', () => ctx.slots.register({ name: 'tool.call.toolview', key: 'realtime_voice_call' }, ToolCard))
        ctx.inject(['commandUi'], sub => {
          sub.effect(() => sub.commandUi.register({ name: 'call', label: () => `打电话给${personaName()}`, description: () => '打开实时语音通话', available: () => true, ui: { kind: 'action', run: () => openCallPage() } }))
        })

        // The sidebar copies the label when an entry registers, so re-register when the persona name changes.
        let disposePanel = null, panelName = ''
        const registerPanel = () => {
          const name = personaName()
          if (!alive || name === panelName) return
          panelName = name
          disposePanel?.()
          disposePanel = ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({ name: 'sidebar.panellist', id: PANEL, order: 30, label: () => `打电话给${name}` }, PhoneIcon))
        }
        registerPanel()
        ctx.effect(() => app.subscribe(registerPanel))
        loadState().catch(() => {})
      },
    }
  },
})
