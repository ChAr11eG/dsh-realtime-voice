// /api/realtime-voice/* HTTP routes and the /realtime-voice/ws call upgrade.
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { Readable } from 'node:stream'
import { WebSocketServer } from 'ws'
import { errorDetail } from './engine.mjs'
import { language } from './text.mjs'

const PREFIX = '/api/realtime-voice'
const JSON_LIMIT = 256 * 1024
const UPLOAD_LIMIT = 30 * 1024 * 1024

const httpError = (status, detail) => Object.assign(new Error(detail), { status })

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    req.on('data', chunk => {
      size += chunk.length
      if (size > limit) {
        reject(httpError(413, '请求体太大'))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', reject)
  })
}

async function readJson(req) {
  const body = await readBody(req, JSON_LIMIT)
  if (!body.length) return {}
  try {
    return JSON.parse(body.toString('utf8'))
  } catch {
    throw httpError(400, '请求体不是合法的 JSON')
  }
}

function sendJson(res, status, value) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' })
  res.end(JSON.stringify(value))
}

/** Copy an engine reply (status, content type, body) to the browser. */
async function pipeEngine(res, reply) {
  if (!reply.ok) return sendJson(res, reply.status, { detail: await errorDetail(reply) })
  res.writeHead(reply.status, {
    'content-type': reply.headers.get('content-type') ?? 'application/octet-stream',
    'cache-control': 'no-store',
  })
  Readable.fromWeb(reply.body).pipe(res)
}

/** 24 kHz mono PCM16 → WAV. */
function wav(pcm, rate = 24000) {
  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + pcm.length, 4)
  header.write('WAVEfmt ', 8)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(rate, 24)
  header.writeUInt32LE(rate * 2, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(pcm.length, 40)
  return Buffer.concat([header, pcm])
}

function listCalls(dataDir) {
  const dir = join(dataDir, 'calls')
  let files = []
  try {
    files = readdirSync(dir).filter(f => f.endsWith('.json'))
  } catch {
    return [] // no calls yet
  }
  const calls = []
  for (const file of files) {
    try {
      const c = JSON.parse(readFileSync(join(dir, file), 'utf8'))
      const users = c.messages.filter(m => m.role === 'user')
      calls.push({
        id: c.id, started: c.started, ended: c.ended, persona: c.persona, topic: c.topic,
        turns: users.length, preview: (users[0]?.text ?? c.messages[0]?.text ?? '').slice(0, 60),
      })
    } catch {
      // skip an unreadable record
    }
  }
  return calls.sort((a, b) => b.started.localeCompare(a.started)).slice(0, 50)
}

/**
 * Register the routes on `sub` (a context with webServer + connection).
 * `app` = { settings, engine, llm, dataDir, state(), startCall(ws) }.
 */
export function registerRoutes(sub, app) {
  const { settings, engine, dataDir } = app
  const rejection = req => sub.connection.requestRejection(req)

  const engineReply = async (method, path, options) => {
    await engine.start()
    engine.armIdle()
    return engine.request(method, path, options)
  }

  async function handle(req, res) {
    const url = new URL(req.url, 'http://x')
    const path = url.pathname.slice(PREFIX.length) || '/'
    const route = `${req.method} ${path}`
    const query = url.search

    switch (route) {
      case 'GET /state':
        return sendJson(res, 200, await app.state())
      case 'POST /settings': {
        const patch = await readJson(req)
        try {
          settings.update(patch)
        } catch (error) {
          throw httpError(400, error.message)
        }
        engine.applySettings().catch(() => {}) // engine errors show up in its state
        return sendJson(res, 200, await app.state())
      }
      case 'POST /secrets': {
        const { name, value } = await readJson(req)
        try {
          settings.setSecret(name, value)
        } catch (error) {
          throw httpError(400, error.message)
        }
        engine.applySettings().catch(() => {})
        return sendJson(res, 200, await app.state())
      }
      case 'POST /engine/start':
        await engine.start().catch(() => {}) // the failure is in State.engine
        return sendJson(res, 200, await app.state())
      case 'POST /engine/stop':
        await engine.stop()
        return sendJson(res, 200, await app.state())
      case 'GET /voices':
      case 'GET /tts/voices':
        return pipeEngine(res, await engineReply('GET', path + query))
      case 'GET /voices/audio':
        return pipeEngine(res, await engineReply('GET', path + query))
      case 'POST /voices/upload': {
        const body = await readBody(req, UPLOAD_LIMIT)
        return pipeEngine(res, await engineReply('POST', path + query, {
          body, headers: { 'content-type': req.headers['content-type'] ?? 'application/octet-stream' },
        }))
      }
      case 'POST /voices/design':
      case 'POST /voices/delete':
        return pipeEngine(res, await engineReply('POST', path, { json: await readJson(req) }))
      case 'POST /tts/preview': {
        const { text, provider = null, voice = null } = await readJson(req)
        if (typeof text !== 'string' || !text.trim()) throw httpError(400, '试听文本不能为空')
        const reply = await engineReply('POST', '/tts', { json: { text, language: language(text), provider, voice } })
        if (!reply.ok) return sendJson(res, reply.status, { detail: await errorDetail(reply) })
        const pcm = Buffer.from(await reply.arrayBuffer())
        res.writeHead(200, { 'content-type': 'audio/wav', 'cache-control': 'no-store' })
        return res.end(wav(pcm.subarray(0, pcm.length - (pcm.length % 2))))
      }
      case 'GET /calls':
        return sendJson(res, 200, { calls: listCalls(dataDir) })
      case 'GET /calls/get': {
        const id = url.searchParams.get('id') ?? ''
        if (!/^[\w-]+$/.test(id)) throw httpError(400, '通话 id 无效')
        try {
          return sendJson(res, 200, JSON.parse(readFileSync(join(dataDir, 'calls', `${id}.json`), 'utf8')))
        } catch {
          throw httpError(404, '没有这条通话记录')
        }
      }
      default:
        throw httpError(404, `未知接口：${route}`)
    }
  }

  sub.effect(() => sub.webServer.register({
    kind: 'prefix',
    path: PREFIX,
    async handler(req, res) {
      const status = rejection(req)
      if (status !== undefined) return sendJson(res, status, { detail: status === 401 ? '未登录' : '禁止访问' })
      try {
        await handle(req, res)
      } catch (error) {
        if (res.headersSent) return res.destroy()
        // engine fetch failures (TypeError) and anything without a status are server errors
        sendJson(res, error.status ?? 500, { detail: error.message })
      }
    },
  }), 'realtime-voice http')

  const wss = new WebSocketServer({ noServer: true, maxPayload: 1024 * 1024 })
  sub.effect(() => sub.webServer.registerUpgrade({
    path: '/realtime-voice/ws',
    handler(req, socket, head) {
      const status = rejection(req)
      if (status !== undefined) {
        socket.end(`HTTP/1.1 ${status} ${status === 401 ? 'Unauthorized' : 'Forbidden'}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`)
        return
      }
      wss.handleUpgrade(req, socket, head, ws => app.startCall(ws))
    },
  }), 'realtime-voice ws')
  sub.effect(() => () => {
    for (const client of wss.clients) client.terminate()
    wss.close()
  }, 'realtime-voice wss')
}
