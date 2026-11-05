// Pulls the demo-recording data (captions, speaker roles, chapter markers,
// audio file names) out of a saved copy of block.xyz/dessa.
// The page's JavaScript is never executed: object literals are read with a
// small data-only parser that rejects anything that is not plain data.
// Usage: node extract_dessa_page.mjs <dessa.html> <out.json>
import { readFileSync, writeFileSync } from 'node:fs'

const [, , htmlPath, outPath] = process.argv
const html = readFileSync(htmlPath, 'utf8')

// Parses one JS literal (object/array/string/number/bool) starting at `i`.
// Returns [value, nextIndex]. Throws on identifiers, calls, spreads, etc.
function parseLiteral(s, i) {
  const ws = () => { while (/\s/.test(s[i])) i++ }
  function value() {
    ws()
    const c = s[i]
    if (c === '{') return object()
    if (c === '[') return array()
    if (c === '"' || c === "'") return string()
    if (s.startsWith('!0', i)) { i += 2; return true }
    if (s.startsWith('!1', i)) { i += 2; return false }
    if (s.startsWith('true', i)) { i += 4; return true }
    if (s.startsWith('false', i)) { i += 5; return false }
    if (s.startsWith('null', i)) { i += 4; return null }
    if (s.startsWith('void 0', i)) { i += 6; return undefined }
    const m = /^-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/i.exec(s.slice(i, i + 40))
    if (m) { i += m[0].length; return Number(m[0]) }
    throw new Error(`not plain data at ${i}: ${s.slice(i, i + 40)}`)
  }
  function string() {
    const q = s[i++]; let out = ''
    while (s[i] !== q) {
      if (s[i] === '\\') {
        const n = s[i + 1]
        if (n === 'u') { out += String.fromCharCode(parseInt(s.slice(i + 2, i + 6), 16)); i += 6; continue }
        if (n === 'x') { out += String.fromCharCode(parseInt(s.slice(i + 2, i + 4), 16)); i += 4; continue }
        out += { n: '\n', t: '\t', r: '\r' }[n] ?? n; i += 2; continue
      }
      out += s[i++]
    }
    i++; return out
  }
  function key() {
    ws()
    if (s[i] === '"' || s[i] === "'") return string()
    const m = /^[A-Za-z_$][\w$-]*/.exec(s.slice(i, i + 80))
    if (!m) throw new Error(`bad key at ${i}: ${s.slice(i, i + 40)}`)
    i += m[0].length; return m[0]
  }
  function object() {
    i++; const o = {}; ws()
    if (s[i] === '}') { i++; return o }
    for (;;) {
      const k = key(); ws()
      if (s[i] !== ':') throw new Error(`expected : at ${i}: ${s.slice(i, i + 40)}`)
      i++; o[k] = value(); ws()
      if (s[i] === ',') { i++; continue }
      if (s[i] === '}') { i++; return o }
      throw new Error(`expected , or } at ${i}: ${s.slice(i, i + 40)}`)
    }
  }
  function array() {
    i++; const a = []; ws()
    if (s[i] === ']') { i++; return a }
    for (;;) {
      a.push(value()); ws()
      if (s[i] === ',') { i++; continue }
      if (s[i] === ']') { i++; return a }
      throw new Error(`expected , or ] at ${i}: ${s.slice(i, i + 40)}`)
    }
  }
  const v = value()
  return [v, i]
}

// Parses the object literal that starts at the first `{` of `marker`.
const literalAt = (marker) => {
  const at = html.indexOf(marker)
  if (at < 0) throw new Error(`marker not found: ${marker}`)
  return parseLiteral(html, html.indexOf('{', at))[0]
}

// 1. Main recordings: each is its own `name={source:"..."}` object.
const main = {}
for (const m of html.matchAll(/([A-Za-z_$][\w$]*)=\{source:"([^"]+)"/g)) {
  const obj = parseLiteral(html, m.index + m[1].length + 1)[0]
  main[obj.source] = obj
}
// Map the page ids to them (from the N1 table on the page).
const mainIds = { 'ai-release': 'AI_release_convo.wav', 'coffee-ding': 'coffe_ding.wav', 'life-advice': 'lifeadvice.wav',
  anniversary: 'anniversary.wav', 'dc-comics': 'DC_comics.wav', 'marc-dessa': 'marcdessa.wav', 'self-play': 'self_play_000006_.wav' }

// 2. "More recordings" table.
const more = literalAt('x1={"more-artist-dessa"')

// 3. Speaker identities and chapter markers.
const who = literalAt('wt={"ai-release":{')
const markersAt = html.lastIndexOf('={', html.indexOf('anniversary:[[13,22,"Excitement"'))
const markers = parseLiteral(html, markersAt + 1)[0]

const recordings = []
const add = (id, obj) => {
  const files = obj.tracks.map((t) => t.src.split('/').pop().replace(/\.wav$/, '.flac'))
  const w = who[id] || {}
  // Page rule: track 1 is labelled Human (or the named other party), track 2 is always Dessa.
  let speaker1 = w.label || (w.textOnly ? w.name : w.otherShape !== undefined ? 'Dessa' : 'Human')
  if (id === 'more-gpt-live-sample') speaker1 = 'GPT-Live-1' // shown on the page as "Dessa in conversation with GPT-Live-1"
  recordings.push({ id, source: obj.source, duration: obj.duration, files, speaker1, speaker2: 'Dessa',
    markers: (markers[id] || (id === 'more-gpt-live-sample' ? markers['gpt-live'] : null) || []).map(([start, end, title, note]) => ({ start, end, title, note })),
    captions: obj.captions || [] })
}
for (const [id, src] of Object.entries(mainIds)) if (main[src]) add(id, main[src])
for (const [id, obj] of Object.entries(more)) add(id, obj)

writeFileSync(outPath, JSON.stringify({ extractedFrom: 'https://block.xyz/dessa', recordings }, null, 1))
for (const r of recordings) console.log(`${r.id.padEnd(32)} ${r.duration.toFixed(1).padStart(6)}s  s1=${r.speaker1.padEnd(18)} captions=${String(r.captions.length).padStart(3)} markers=${r.markers.length} files=${r.files.join(',')}`)
