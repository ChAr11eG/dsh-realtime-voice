// Supervisor and HTTP client for the Python speech engine (engine/server.py).
import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { createWriteStream, existsSync } from 'node:fs'
import { join } from 'node:path'
import { WebSocket } from 'ws'

const LOG_LINES = 80

async function errorDetail(response) {
  const text = await response.text().catch(() => '')
  try {
    return JSON.parse(text).detail ?? text
  } catch {
    return text || `HTTP ${response.status}` // not JSON: show the raw text
  }
}

/** One engine process at a time; state: stopped | starting | warming | ready | error. */
export class Engine {
  constructor({ repoDir, dataDir, settings }) {
    this.repoDir = repoDir
    this.dataDir = dataDir
    this.settings = settings
    this.state = 'stopped'
    this.error = null
    this.health = null
    this.log = []
    this.child = null
    this.port = null
    this.token = null
    this.up = null // promise: process answers /health and has its config
    this.warming = null // promise: /warm finished
    this.busy = false // a call is using the engine
    this.idleTimer = null
    this.listeners = new Set()
    this.warmKey = null
  }

  /** Subscribe to state changes; returns the unsubscribe function. */
  listen(fn) {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  setState(state, error = null) {
    this.state = state
    this.error = error
    for (const fn of this.listeners) fn(state, error)
  }

  snapshot() {
    return { state: this.state, error: this.error, health: this.health, log: this.log }
  }

  python() {
    const configured = this.settings.get().engine.python
    if (configured) return configured
    const venv = join(this.repoDir, '.venv', 'bin', 'python')
    return existsSync(venv) ? venv : 'python3'
  }

  /** The whole engine config object (SPEC §1), secrets included. */
  config() {
    const s = this.settings.get()
    const secret = name => this.settings.secret(name)
    return {
      asr: {
        provider: s.asr.provider, language: s.asr.language, partials: s.asr.partials,
        partialInterval: s.asr.partialInterval, qwen3: s.asr.qwen3,
        openai: { ...s.asr.openai, apiKey: secret('asrOpenaiKey') },
      },
      turn: s.turn,
      tts: {
        provider: s.tts.provider, breeze: s.tts.breeze, edge: s.tts.edge,
        openai: { ...s.tts.openai, apiKey: secret('ttsOpenaiKey') },
      },
    }
  }

  // Changing any of these loads different models, so a running engine re-warms.
  providerKey() {
    const { asr, tts } = this.settings.get()
    return JSON.stringify([asr.provider, asr.qwen3, tts.provider, tts.breeze])
  }

  appendLog(text) {
    for (const line of text.split('\n')) {
      if (!line) continue
      this.log.push(line)
      if (this.log.length > LOG_LINES) this.log.shift()
    }
  }

  /** Start the process if needed; resolves once it answers /health. Warming continues in the background. */
  start() {
    if (this.up) return this.up
    this.clearIdle()
    this.port = this.settings.get().engine.port
    this.token = randomBytes(24).toString('hex')
    this.health = null
    this.setState('starting')
    const file = createWriteStream(join(this.dataDir, 'engine.log'), { flags: 'a', mode: 0o600 })
    file.on('error', () => {}) // a log write failure must never take down DSH
    // The token goes through the environment, not argv (argv is readable by every local user).
    const child = spawn(this.python(), [
      '-u', 'engine/server.py', '--port', String(this.port), '--data-dir', this.dataDir,
    ], { cwd: this.repoDir, detached: true, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, RV_TOKEN: this.token } })
    this.child = child
    const exited = new Promise(resolve => {
      child.once('error', error => resolve(`无法启动引擎（${this.python()}）：${error.message}`))
      child.once('exit', (code, signal) => resolve(`引擎已退出（${signal ?? code}）`))
    })
    const onData = chunk => {
      if (!file.writableEnded) file.write(chunk)
      this.appendLog(chunk.toString('utf8'))
    }
    child.stdout.on('data', onData)
    child.stderr.on('data', onData)
    child.once('close', () => file.end()) // after stdio drained; 'exit' can come before the last output
    exited.then(reason => {
      // The engine is gone; make sure the Breeze worker in its process group goes too.
      try { process.kill(-child.pid, 'SIGKILL') } catch { /* group already gone */ }
      if (this.child !== child) return
      this.child = null
      this.up = this.warming = null
      this.appendLog(reason)
      if (this.state !== 'stopped') this.setState('error', reason)
    })
    this.up = (async () => {
      const deadline = Date.now() + 60_000
      for (;;) {
        const reason = await Promise.race([exited, new Promise(r => setTimeout(r, 300))])
        if (reason) throw new Error(reason)
        try {
          const res = await this.request('GET', '/health', { timeout: 2000 })
          if (res.ok) break
        } catch {
          // not listening yet
        }
        if (Date.now() > deadline) throw new Error('引擎 60 秒内没有响应 /health')
      }
      this.health = await this.json('POST', '/config', this.config())
      this.warm()
      this.armIdle()
    })()
    this.up.catch(error => {
      if (this.state === 'stopped') return
      if (this.child === child) this.kill()
      this.setState('error', error.message)
    })
    return this.up
  }

  /** POST /warm; state warming until it returns. */
  warm() {
    this.warmKey = this.providerKey()
    this.setState('warming')
    const run = this.warming = this.json('POST', '/warm', undefined, { timeout: 400_000 }).then(health => {
      this.health = health
      if (this.warming === run) this.setState('ready')
    }, error => {
      if (this.warming === run && this.child) this.setState('error', error.message)
      throw error
    })
    run.catch(() => {}) // reported through state
    return run
  }

  /** Start if needed and wait until warmed; `onState` sees each state change meanwhile. */
  async ensureReady(onState) {
    const off = onState ? this.listen(onState) : () => {}
    try {
      onState?.(this.state, this.error)
      await this.start()
      if (this.state === 'error' && this.child) this.warm() // retry a failed warm-up
      await this.warming
    } finally {
      off()
    }
  }

  /** Push the current config to a running engine; re-warm if a provider changed. */
  async applySettings() {
    if (!this.child || !this.up) return
    await this.up
    this.health = await this.json('POST', '/config', this.config())
    if (this.providerKey() !== this.warmKey) this.warm()
    this.armIdle()
  }

  kill() {
    const child = this.child
    if (!child) return Promise.resolve()
    this.child = null
    this.up = this.warming = null
    return new Promise(resolve => {
      if (!child.pid) return resolve()
      const killGroup = () => { try { process.kill(-child.pid, 'SIGKILL') } catch { /* group already gone */ } }
      if (child.exitCode !== null || child.signalCode !== null) { killGroup(); return resolve() }
      const timer = setTimeout(killGroup, 10_000)
      child.once('exit', () => { clearTimeout(timer); killGroup(); resolve() })
      try { process.kill(-child.pid, 'SIGTERM') } catch { /* group already gone */ }
    })
  }

  async stop() {
    this.clearIdle()
    this.health = null
    this.setState('stopped')
    await this.kill()
  }

  clearIdle() {
    clearTimeout(this.idleTimer)
    this.idleTimer = null
  }

  /** (Re)start the idle countdown unless a call is running. */
  armIdle() {
    this.clearIdle()
    const minutes = this.settings.get().engine.idleMinutes
    if (this.busy || !this.child || !minutes) return
    this.idleTimer = setTimeout(() => {
      if (!this.busy) this.stop()
    }, minutes * 60_000)
    this.idleTimer.unref()
  }

  setBusy(busy) {
    this.busy = busy
    if (busy) this.clearIdle()
    else this.armIdle()
  }

  /** Raw fetch to the engine with the token header. */
  request(method, path, { json, body, headers = {}, signal, timeout } = {}) {
    if (json !== undefined) {
      body = JSON.stringify(json)
      headers = { ...headers, 'content-type': 'application/json' }
    }
    const signals = [signal, timeout && AbortSignal.timeout(timeout)].filter(Boolean)
    return fetch(`http://127.0.0.1:${this.port}${path}`, {
      method, body, duplex: body ? 'half' : undefined,
      headers: { ...headers, 'x-rv-token': this.token },
      signal: signals.length ? AbortSignal.any(signals) : undefined,
    })
  }

  /** JSON request; throws Error(detail) on a non-2xx reply. */
  async json(method, path, json, options = {}) {
    const res = await this.request(method, path, { ...options, json })
    if (!res.ok) throw Object.assign(new Error(await errorDetail(res)), { status: res.status })
    return res.json()
  }

  /** New WebSocket to the engine's /asr stream. */
  asrSocket() {
    return new WebSocket(`ws://127.0.0.1:${this.port}/asr`, { headers: { 'x-rv-token': this.token } })
  }
}

export { errorDetail }
