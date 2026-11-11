// DSH host entry: settings, engine supervisor, call WebSocket, HTTP API and the call tool.
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { CallSession } from './host/call.mjs'
import { Engine } from './host/engine.mjs'
import { createLlm } from './host/llm.mjs'
import { registerRoutes } from './host/routes.mjs'
import { createSettings } from './host/settings.mjs'

export const name = 'realtime-voice'
export const inject = ['tools']

const repoDir = dirname(fileURLToPath(import.meta.url))

export async function apply(ctx, config = {}) {
  const dataDir = config.dataDir
    || join(process.env.DSH_HOME || join(homedir(), '.dsh'), 'realtime-voice')
  const settings = createSettings(dataDir, console)
  const engine = new Engine({ repoDir, dataDir, settings })
  const llm = createLlm(ctx, settings)
  let pendingCall = null // { topic, at } set by the tool, consumed by the next call
  let call = null

  async function state() {
    return {
      settings: settings.get(), secrets: settings.secretFlags(), engine: engine.snapshot(),
      llmCatalog: await llm.catalog(), pendingCall, inCall: call !== null, repoDir, dataDir,
    }
  }

  function startCall(ws) {
    if (call) {
      ws.send(JSON.stringify({ type: 'error', detail: '已经在通话中' }))
      ws.close(1008, 'busy')
      return
    }
    const session = call = new CallSession({ ws, engine, llm, settings, dataDir, topic: pendingCall?.topic })
    engine.setBusy(true)
    session.ended.then(() => {
      if (call !== session) return
      call = null
      pendingCall = null
      engine.setBusy(false)
    })
    session.run()
  }

  ctx.tools.register({
    name: 'realtime_voice_call',
    description: `打开与${settings.get().persona.name}的实时语音通话。当用户想用语音聊天、打电话给这个角色（例如“打电话给${settings.get().persona.name}”）时调用。`
      + ' Opens a realtime voice call with the persona. Call it when the user wants to talk by voice or phone the persona.',
    parameters: {
      type: 'object',
      properties: { topic: { type: 'string', description: '用户想聊的话题（可选） / What the user wants to talk about (optional)' } },
      additionalProperties: false,
    },
    timeoutMs: 10000,
    output: {
      schema: {
        type: 'object',
        properties: { ok: { type: 'boolean' }, persona: { type: 'string' }, topic: { type: 'string' } },
        required: ['ok', 'persona', 'topic'],
        additionalProperties: false,
      },
      render: (_args, value) => [{
        type: 'text',
        text: `已在 DSH 中打开与${value.persona}的通话页面，用户现在可以直接说话。`
          + ` The call page with ${value.persona} is open in DSH; the user can talk now.`,
      }],
    },
    async execute(args) {
      const topic = typeof args?.topic === 'string' ? args.topic.trim() : ''
      pendingCall = { topic, at: new Date().toISOString() }
      return { ok: true, persona: settings.get().persona.name, topic }
    },
  })

  ctx.inject(['webServer', 'connection'], sub => {
    registerRoutes(sub, { settings, engine, llm, dataDir, state, startCall })
  })

  // One ordered teardown: hang up, then stop the engine process group.
  ctx.effect(() => async () => {
    call?.close('shutdown')
    await engine.stop()
  }, 'realtime-voice engine')
}
