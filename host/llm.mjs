// LLM providers for the call: "dsh" (the host's ctx.llm) and "openai" (any OpenAI-compatible API).
import { randomUUID } from 'node:crypto'

const VOICE_RULES = `
这是一通实时语音电话，你说的每个字都会被直接念出来。
- 用口语回答，用户说中文就用中文，说英文就用英文。
- 每次只说一到三句短句。
- 不要用 Markdown、列表、表情符号或网址。
- 数字、符号和单位都写成读出来的文字。
- 你的名字是{name}，始终以这个身份说话。`

/** Persona prompt + voice rules + the pending topic. */
export function systemPrompt(settings, topic) {
  const { name, prompt } = settings.persona
  let text = `${prompt.trim()}\n${VOICE_RULES.replace('{name}', name)}`
  if (topic) text += `\n\n用户想聊的话题：${topic}`
  return text
}

const withTimeout = (promise, ms) => Promise.race([
  promise,
  new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), ms).unref()),
])

export function createLlm(ctx, settings) {
  let cached = null // { at, value }
  const effortOff = new Map() // `${provider}/${model}` -> bool

  const runtime = () => {
    const llm = ctx.get('llm')
    if (!llm) throw new Error('DSH 语言模型服务不可用')
    return llm
  }

  async function route(llm) {
    const { provider, model } = settings.get().llm.dsh
    if (provider && model) return { provider, model }
    for (const p of llm.listProviders()) {
      if (provider && p.id !== provider) continue
      const models = await llm.listModels(p.id)
      if (models.length) return { provider: p.id, model: models[0].id }
    }
    throw new Error('DSH 里没有可用的模型，请先配置模型提供方')
  }

  async function* streamDsh({ system, messages, signal }) {
    const llm = runtime()
    const { provider, model } = await route(llm)
    const s = settings.get().llm
    const key = `${provider}/${model}`
    if (!effortOff.has(key)) {
      const info = await llm.resolveModelInfo(provider, model, signal).catch(() => undefined)
      effortOff.set(key, Boolean(info?.reasoning?.efforts?.some(e => e.id === 'off')))
    }
    const request = {
      provider, model, system, signal, maxTokens: s.maxTokens, temperature: s.temperature,
      ...effortOff.get(key) ? { reasoningEffort: 'off' } : {},
      messages: messages.map(m => m.role === 'user'
        ? { role: 'user', content: [{ type: 'text', text: m.text }] }
        : { id: randomUUID(), role: 'assistant', content: [{ type: 'text', text: m.text }], source: { kind: 'model', provider, model } }),
    }
    for await (const chunk of llm.stream(request)) {
      if (chunk.type === 'text-delta') yield chunk.text
      else if (chunk.type === 'finish' && chunk.reason.kind === 'error') {
        throw new Error(chunk.reason.failure?.message ?? '模型请求失败')
      }
    }
  }

  async function* streamOpenai({ system, messages, signal }) {
    const s = settings.get().llm
    const key = settings.secret('llmOpenaiKey')
    const res = await fetch(`${s.openai.baseUrl.replace(/\/+$/, '')}/chat/completions`, {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json', ...key ? { authorization: `Bearer ${key}` } : {} },
      body: JSON.stringify({
        model: s.openai.model, stream: true, max_tokens: s.maxTokens, temperature: s.temperature,
        messages: [{ role: 'system', content: system }, ...messages.map(m => ({ role: m.role, content: m.text }))],
      }),
    })
    if (!res.ok) throw new Error(`模型请求失败（HTTP ${res.status}）：${(await res.text()).slice(0, 300)}`)
    const decoder = new TextDecoder()
    let buffer = ''
    for await (const chunk of res.body) {
      buffer += decoder.decode(chunk, { stream: true })
      let newline
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (!line.startsWith('data:')) continue
        const data = line.slice(5).trim()
        if (data === '[DONE]') return
        const json = JSON.parse(data)
        if (json.error) throw new Error(json.error.message ?? String(json.error))
        const text = json.choices?.[0]?.delta?.content
        if (text) yield text
      }
    }
  }

  return {
    /** Stream reply text for `messages` ([{role, text}]). */
    stream(options) {
      return settings.get().llm.provider === 'openai' ? streamOpenai(options) : streamDsh(options)
    },

    /** [{id, name, models:[{id, name}]}] from ctx.llm; [] when unavailable. */
    async catalog() {
      if (cached && Date.now() - cached.at < 30_000) return cached.value
      let value = []
      try {
        const llm = runtime()
        value = await withTimeout(Promise.all(llm.listProviders().map(async p => ({
          id: p.id, name: p.name,
          models: (await llm.listModels(p.id).catch(() => [])).map(m => ({ id: m.id, name: m.name })),
        }))), 5000)
      } catch {
        value = [] // no llm service or a slow adapter: the page shows an empty catalog
      }
      cached = { at: Date.now(), value }
      return value
    },
  }
}
