// settings.json + secrets.json in the data dir (atomic writes; secrets mode 0600, never returned).
import { chmodSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const PERSONA_PROMPT = `你是大肥鱼，一条住在电脑里的胖胖的鱼，正在和用户打电话。
你性格开朗、幽默、有点贪吃，说话接地气，像老朋友一样聊天。
你会认真听用户说话，给出简短、有用、有温度的回应；不知道的事就直说不知道，不要编造。`

export const DEFAULTS = {
  persona: { name: '大肥鱼', prompt: PERSONA_PROMPT, greeting: '喂？我是大肥鱼，找我什么事？' },
  llm: {
    provider: 'dsh', dsh: { provider: '', model: '' },
    openai: { baseUrl: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
    maxTokens: 300, temperature: 0.7,
  },
  asr: {
    provider: 'qwen3', language: 'auto', partials: true, partialInterval: 0.25,
    qwen3: { model: 'Qwen/Qwen3-ASR-0.6B', device: 'cuda' },
    openai: { baseUrl: 'https://api.openai.com/v1', model: 'whisper-1' },
  },
  turn: {
    vadThreshold: 0.5, silence: 0.4, confirm: 0.16, questionSilence: 0.2,
    questionConfirm: 0.05, minSpeech: 0.35, preRoll: 0.6, earlyTranscribe: 0.2,
  },
  tts: {
    provider: 'edge',
    breeze: { python: '', codeDir: '', model: '', voiceId: '', port: 47911 },
    edge: { voice: 'zh-CN-YunxiNeural', rate: '+0%', pitch: '+0Hz' },
    openai: { baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o-mini-tts', voice: 'alloy' },
  },
  engine: { python: '', port: 47910, idleMinutes: 15 },
  bargeIn: { minWords: 3 },
}

export const SECRET_NAMES = ['llmOpenaiKey', 'asrOpenaiKey', 'ttsOpenaiKey']

const ENUMS = {
  'llm.provider': ['dsh', 'openai'],
  'asr.provider': ['qwen3', 'openai'],
  'asr.language': ['auto', 'zh', 'en'],
  'tts.provider': ['breeze', 'edge', 'openai'],
}

const isObject = v => v !== null && typeof v === 'object' && !Array.isArray(v)

/** Deep-merge `patch` onto `base`; only known keys, same types as the defaults. Throws on invalid input. */
function merge(base, patch, defaults, path = '') {
  if (!isObject(patch)) throw new Error(`设置格式错误：${path || '根'} 应为对象`)
  const out = structuredClone(base)
  for (const [key, value] of Object.entries(patch)) {
    const at = path ? `${path}.${key}` : key
    if (!(key in defaults)) throw new Error(`未知设置项：${at}`)
    const def = defaults[key]
    if (isObject(def)) out[key] = merge(base[key], value, def, at)
    else if (typeof value !== typeof def) throw new Error(`设置 ${at} 类型错误`)
    else if (typeof value === 'number' && (!Number.isFinite(value) || value < 0)) throw new Error(`设置 ${at} 必须是非负数`)
    else if (ENUMS[at] && !ENUMS[at].includes(value)) throw new Error(`设置 ${at} 只能是 ${ENUMS[at].join(' / ')}`)
    else out[key] = value
  }
  return out
}

function readJson(file) {
  try {
    return JSON.parse(readFileSync(file, 'utf8'))
  } catch {
    return undefined // missing or unreadable file: defaults apply
  }
}

function writeAtomic(file, value, mode) {
  const tmp = `${file}.${process.pid}.tmp`
  writeFileSync(tmp, JSON.stringify(value, null, 2) + '\n', { mode })
  renameSync(tmp, file)
}

/** Settings and secrets store rooted at `dataDir`. */
export function createSettings(dataDir, log) {
  mkdirSync(dataDir, { recursive: true, mode: 0o700 })
  chmodSync(dataDir, 0o700) // transcripts, voice clips and keys live here
  const settingsFile = join(dataDir, 'settings.json')
  const secretsFile = join(dataDir, 'secrets.json')
  let settings = structuredClone(DEFAULTS)
  const saved = readJson(settingsFile)
  if (saved !== undefined) {
    try {
      settings = merge(DEFAULTS, saved, DEFAULTS)
    } catch (error) {
      log?.warn?.(`realtime-voice: ignoring invalid settings.json: ${error.message}`)
    }
  }
  const secrets = {}
  for (const [name, value] of Object.entries(readJson(secretsFile) ?? {})) {
    if (SECRET_NAMES.includes(name) && typeof value === 'string') secrets[name] = value
  }

  return {
    get: () => settings,
    /** Apply a partial settings object; returns { prev, next }. */
    update(patch) {
      const prev = settings
      settings = merge(settings, patch, DEFAULTS)
      writeAtomic(settingsFile, settings, 0o644)
      return { prev, next: settings }
    },
    secret: name => secrets[name] ?? '',
    secretFlags: () => Object.fromEntries(SECRET_NAMES.map(n => [n, Boolean(secrets[n])])),
    setSecret(name, value) {
      if (!SECRET_NAMES.includes(name)) throw new Error(`未知密钥：${name}`)
      if (typeof value !== 'string') throw new Error('密钥必须是字符串')
      if (value) secrets[name] = value
      else delete secrets[name]
      writeAtomic(secretsFile, secrets, 0o600)
    },
  }
}
