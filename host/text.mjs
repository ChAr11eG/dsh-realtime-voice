// Text helpers for the spoken reply: phrase splitting,
// barge-in word counting and speech cleanup.

const FIRST_TOKENS = new Set('.!?,;:\n-。、，！？；：')
const NEXT_HARD = new Set('.!?;\n。！？；')
const NEXT_SOFT = new Set(',，、：')
const ALNUM = /[\p{L}\p{N}]/u

const isHan = char => char >= '㐀' && char <= '鿿'
const isAscii = char => char.charCodeAt(0) < 128

// ASCII punctuation splits only before whitespace, never at the end of the buffer
// or inside 12.5 / twenty-five.
function asciiHold(text, char, end) {
  return isAscii(char) && char !== '\n' && (end === text.length || !/\s/.test(text[end]))
}

/** End index of the first phrase worth sending to TTS, or null. */
export function firstPhrase(text, maxLen = 120, minWeight = 10) {
  let weight = 0
  let count = 0
  let end = 0
  for (const char of text) {
    if (++count > maxLen) break
    end += char.length
    if (ALNUM.test(char)) weight += isHan(char) ? 2.5 : 1
    if (FIRST_TOKENS.has(char) && weight >= minWeight) {
      if (asciiHold(text, char, end)) continue
      return end
    }
  }
  return null
}

/** End index of a later phrase: sentence ends at weight 10, clause ends at weight 40, forced at 200 chars. */
export function nextPhrase(text, force = 200) {
  let weight = 0
  let count = 0
  let end = 0
  for (const char of text) {
    end += char.length
    if (ALNUM.test(char)) weight += isHan(char) ? 2.5 : 1
    if ((NEXT_HARD.has(char) && weight >= 10) || (NEXT_SOFT.has(char) && weight >= 40)) {
      if (!asciiHold(text, char, end)) return end
    }
    if (++count >= force) return end
  }
  return null
}

/** Residual echo transcribes as one or two words; real barge-in says more. */
export function substantial(text, minWords) {
  return (text.match(/[一-鿿]/g) ?? []).length >= minWords
    || (text.match(/[A-Za-z0-9']+/g) ?? []).length >= minWords
}

/** Remove complete <think> blocks and an unfinished trailing one. */
export function stripThink(text) {
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/<think>[\s\S]*$/, '')
}

/** Text as it should reach TTS: no reasoning, markdown, emoji or URLs. */
export function cleanSpeech(text) {
  return stripThink(text)
    .replace(/https?:\/\/\S+|www\.\S+/g, '')
    .replace(/[*_#`]/g, '')
    .replace(/[\p{Extended_Pictographic}‍️]/gu, '')
    .replace(/\s+/g, ' ')
    .replace(/(?<=[㐀-鿿])\s+(?=[㐀-鿿])|(?<=[，。！？；：、）」』])\s+|\s+(?=[，。！？；：、（「『])/g, '') // no spaces inside Chinese
    .trim()
}

/** 'zh' if the text has any Han character, else 'en'. */
export function language(text) {
  return /[㐀-鿿]/.test(text) ? 'zh' : 'en'
}
