import {RangeSetBuilder, StateEffect, StateField} from '@codemirror/state';
import {Decoration, DecorationSet, EditorView, keymap, ViewPlugin, ViewUpdate, WidgetType,} from '@codemirror/view';
import {App, Editor, MarkdownView, Notice, Plugin, PluginSettingTab, Setting} from 'obsidian';

/* ------------------------------------------------------------------------------------------------
 * Settings
 * ----------------------------------------------------------------------------------------------*/
interface ScribePilotSettings {
  // Operation mode
  mode: 'offline'|'hybrid'|'cloud';

  // Prediction provider & endpoint
  provider: 'auto'|'ollama'|'llamacpp'|'openai';
  endpoint:
      string;     // e.g. http://localhost:11434 (Ollama), http://localhost:8080
                  // (llama.cpp), https://api.openai.com
  model: string;  // e.g. llama3:8b (Ollama), gguf-name (llama.cpp), gpt-4o-mini
                  // (OpenAI)

  // Auth for cloud providers
  apiKey: string;

  // Prediction UX
  maxTokens: number;
  debounceMs: number;

  // Spelling (SymSpell)
  enableSymSpell: boolean;
  symSpellDistance: 1|2;
  symSpellDictUrl: string;  // URL to "word frequency" dictionary or empty to
                            // use built-in tiny seed
  underlineTypos: boolean;

  // LanguageTool
  enableLanguageTool: boolean;
  ltEndpoint: string;  // e.g. https://api.languagetool.org
  ltLanguage: string;  // e.g. en-US
  ltDebounceMs: number;
}

const DEFAULT_SETTINGS: ScribePilotSettings = {
  mode: 'offline',
  provider: 'auto',
  endpoint: '',
  model: 'llama3:8b',
  apiKey: '',
  maxTokens: 32,
  debounceMs: 180,

  enableSymSpell: true,
  symSpellDistance: 2,
  symSpellDictUrl: '',  // can be set to a frequency list; see README
  underlineTypos: true,

  enableLanguageTool: true,
  ltEndpoint: 'https://api.languagetool.org',
  ltLanguage: 'en-US',
  ltDebounceMs: 700,
};

/* ------------------------------------------------------------------------------------------------
 * Utilities
 * ----------------------------------------------------------------------------------------------*/
function safeJsonParse(line: string): any|null {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

function isPunctuationBoundary(ch: string): boolean {
  return /[.!?]/.test(ch);
}

function lastSentenceFragment(text: string, maxChars = 800): string {
  const slice = text.slice(-maxChars);
  const idx = Math.max(
      slice.lastIndexOf('\n\n'), slice.lastIndexOf('. '),
      slice.lastIndexOf('! '), slice.lastIndexOf('? '));
  return idx >= 0 ? slice.slice(idx + 1) : slice;
}

/* ------------------------------------------------------------------------------------------------
 * SymSpell-lite (embedded)
 *  - Minimal, memory-friendly subset adequate for realtime typo hints
 *  - Uses deletes for candidate generation + frequency ranking
 * ----------------------------------------------------------------------------------------------*/
class SymSpellLite {
  private dictionary = new Map<string, number>();    // word -> frequency
  private deletes = new Map<string, Set<string>>();  // deleteKey -> set(words)
  private maxDistance: 1|2 = 2;

  async loadFromUrl(url: string, maxDistance: 1|2) {
    this.maxDistance = maxDistance;
    const res = await fetch(url);
    if (!res.ok)
      throw new Error(`Failed to load SymSpell dictionary from ${url}`);
    const text = await res.text();
    this.ingestFrequencyList(text);
  }

  loadTinySeed(maxDistance: 1|2) {
    this.maxDistance = maxDistance;
    // A tiny embedded seed so it "works" out of the box.
    const seed = [
      ['the', 23135851162], ['be', 12545825],  ['to', 11765701],
      ['of', 11632336],     ['and', 9797594],  ['a', 9487165],
      ['in', 8469400],      ['that', 5935520], ['have', 4301553],
      ['I', 3991696],       ['it', 3722010],   ['for', 3669033],
      ['not', 3093440],     ['on', 3038986],   ['with', 2971052],
      ['he', 2625957],      ['as', 2483612],   ['you', 2333510],
      ['do', 2127955],      ['at', 2022435],   ['this', 1961200],
      ['but', 1929641],     ['his', 1841598],  ['by', 1799408],
      ['from', 1742060],    ['they', 1686036], ['we', 1651821],
    ] as const;
    for (const [w, f] of seed) this.dictionary.set(w.toLowerCase(), f);
    this.rebuildDeletes();
  }

  private ingestFrequencyList(text: string) {
    // Expect lines: "word freq"
    const lines = text.split(/\r?\n/);
    for (const line of lines) {
      const [w, fStr] = line.trim().split(/\s+/);
      if (!w) continue;
      const f = Number(fStr ?? 1);
      this.dictionary.set(w.toLowerCase(), isFinite(f) ? f : 1);
    }
    this.rebuildDeletes();
  }

  private rebuildDeletes() {
    this.deletes.clear();
    for (const word of this.dictionary.keys()) {
      for (const del of this.generateDeletes(word, this.maxDistance)) {
        if (!this.deletes.has(del)) this.deletes.set(del, new Set());
        this.deletes.get(del)?.add(word);
      }
    }
  }

  private generateDeletes(word: string, distance: number): Iterable<string> {
    const queue = new Set<string>([word]);
    const results = new Set<string>();
    for (let d = 0; d < distance; d++) {
      const next = new Set<string>();
      for (const w of queue) {
        for (let i = 0; i < w.length; i++) {
          const del = w.slice(0, i) + w.slice(i + 1);
          if (!results.has(del)) {
            results.add(del);
            next.add(del);
          }
        }
      }
      for (const n of next) queue.add(n);
    }
    return results;
  }

  private damerauLevenshtein(a: string, b: string): number {
    // Small DL distance for ranking candidates
    const da: Record<string, number> = {};
    const maxDist = this.maxDistance + 1;
    const INF = a.length + b.length;
    const H =
        Array(a.length + 2).fill(null).map(() => Array(b.length + 2).fill(0));
    H[0][0] = INF;
    for (let i = 0; i <= a.length; i++) {
      H[i + 1][0] = INF;
      H[i + 1][1] = i;
    }
    for (let j = 0; j <= b.length; j++) {
      H[0][j + 1] = INF;
      H[1][j + 1] = j;
    }
    for (let i = 1; i <= a.length; i++) {
      let db = 0;
      for (let j = 1; j <= b.length; j++) {
        const i1 = da[b[j - 1]] ?? 0;
        const j1 = db;
        const cost = a[i - 1] === b[j - 1] ? (db = j, 0) : 1;
        H[i + 1][j + 1] = Math.min(
            H[i][j] + cost,
            H[i + 1][j] + 1,
            H[i][j + 1] + 1,
            H[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
        );
      }
      da[a[i - 1]] = i;
      // small cut-off for speed
      if (Math.min(...H[i + 1].slice(1)) > maxDist) return maxDist;
    }
    return H[a.length + 1][b.length + 1];
  }

  /** Returns best suggestion (if any) for a single word. */
  suggest(word: string): {suggestion: string, distance: number}|null {
    const w = word.toLowerCase();
    if (this.dictionary.has(w)) return null;

    // Gather candidates from deletes
    const candidates = new Set<string>();
    for (const del of this.generateDeletes(w, this.maxDistance)) {
      const hits = this.deletes.get(del);
      if (hits)
        for (const h of hits) candidates.add(h);
    }
    if (!candidates.size) return null;

    // Rank by (distance asc, frequency desc, lex asc)
    let best: {w: string, d: number, f: number}|null = null;
    for (const c of candidates) {
      const d = this.damerauLevenshtein(w, c);
      if (d > this.maxDistance) continue;
      const f = this.dictionary.get(c) ?? 1;
      if (!best || d < best.d || (d === best.d && f > best.f) ||
          (d === best.d && f === best.f && c < best.w)) {
        best = {w: c, d, f};
      }
    }
    return best ? {suggestion: best.w, distance: best.d} : null;
  }
}

/* ------------------------------------------------------------------------------------------------
 * Prediction Provider (offline + cloud)
 *  - Keeps your original Ollama/llama.cpp streaming
 *  - Adds OpenAI-compatible cloud chat streaming
 *  - Respects mode: offline | hybrid | cloud
 * ----------------------------------------------------------------------------------------------*/
class StreamPredictor {
  private settings: () => ScribePilotSettings;
  constructor(getSettings: () => ScribePilotSettings) {
    this.settings = getSettings;
  }

  async * predictStream(prompt: string): AsyncGenerator<string, void, unknown> {
    const s = this.settings();

    // Resolve provider + endpoint
    let provider = s.provider;
    const endpoint = s.endpoint;

    const tryOffline = async function*(self: StreamPredictor) {
      let offlineProvider = provider;
      let offlineEndpoint = endpoint;

      // Auto-detect offline servers if needed
      if (offlineProvider === 'auto' || offlineProvider === 'openai') {
        const o = await self.ping('http://localhost:11434');
        if (o) {
          offlineProvider = 'ollama';
          offlineEndpoint = 'http://localhost:11434';
        } else {
          const l = await self.ping('http://localhost:8080');
          if (l) {
            offlineProvider = 'llamacpp';
            offlineEndpoint = 'http://localhost:8080';
          }
        }
      } else {
        if (!offlineEndpoint) {
          offlineEndpoint = offlineProvider === 'ollama' ?
              'http://localhost:11434' :
              'http://localhost:8080';
        }
      }

      if (offlineProvider === 'ollama' && offlineEndpoint) {
        yield*
            self.streamFromOllama(
                offlineEndpoint, s.model, prompt, s.maxTokens);
        return;
      }
      if (offlineProvider === 'llamacpp' && offlineEndpoint) {
        yield*
            self.streamFromLlamaCpp(
                offlineEndpoint, s.model, prompt, s.maxTokens);
        return;
      }
      throw new Error('No local backend detected.');
    }.bind(null, this);

    const tryCloud = async function*(self: StreamPredictor) {
      const cloudEndpoint = endpoint || 'https://api.openai.com';
      if (!s.apiKey) throw new Error('Missing API key for cloud provider.');
      if (provider !== 'openai' && s.mode === 'cloud') {
        // We only implement OpenAI-compatible in this file for simplicity.
        provider = 'openai';
      }
      if (provider === 'openai') {
        yield*
            self.streamFromOpenAI(
                cloudEndpoint, s.model || 'gpt-4o-mini', prompt, s.apiKey,
                s.maxTokens);
        return;
      }
      throw new Error('Unsupported cloud provider.');
    }.bind(null, this);

    try {
      if (s.mode === 'offline') {
        yield* tryOffline();
        return;
      }
      if (s.mode === 'cloud') {
        yield* tryCloud();
        return;
      }
      // hybrid: offline first, then cloud
      try {
        yield* tryOffline();
      } catch {
        yield* tryCloud();
      }
      return;
    } catch (e) {
      // Last resort: if user picked explicit offline provider and it failed,
      // try the other offline quickly
      if (s.mode !== 'cloud' && s.provider !== 'openai') {
        try {
          if (provider === 'ollama') {
            yield*
                this.streamFromLlamaCpp(
                    'http://localhost:8080', s.model, prompt, s.maxTokens);
            return;
          } else {
            yield*
                this.streamFromOllama(
                    'http://localhost:11434', s.model, prompt, s.maxTokens);
            return;
          }
        } catch { /* swallow */
        }
      }
      throw e;
    }
  }

  private async ping(base: string): Promise<boolean> {
    try {
      const ctl = new AbortController();
      const t = setTimeout(() => ctl.abort(), 250);
      const res = await fetch(base, {method: 'GET', signal: ctl.signal});
      clearTimeout(t);
      return res.ok;
    } catch {
      return false;
    }
  }

  // Ollama JSONL streaming
  private async *
      streamFromOllama(
          base: string, model: string, prompt: string, maxTokens: number) {
    const url = `${base.replace(/\/$/, '')}/api/generate`;
    const body =
        {model, prompt, stream: true, options: {num_predict: maxTokens}};
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    if (!res.ok || !res.body) throw new Error(`Ollama error: ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let idx: number;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        const obj = safeJsonParse(line);
        if (!obj) continue;
        if (obj.response) yield obj.response as string;
        if (obj.done) return;
      }
    }
  }

  // llama.cpp streaming: prefer /v1/completions, fallback /completion
  private async *
      streamFromLlamaCpp(
          base: string, model: string, prompt: string, maxTokens: number) {
    const root = base.replace(/\/$/, '');
    let url = `${root}/v1/completions`;
    let body: any = {model, prompt, max_tokens: maxTokens, stream: true};
    let res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });

    if (res.status === 404) {
      url = `${root}/completion`;
      body = {prompt, n_predict: maxTokens, stream: true, cache_prompt: true};
      res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
    }
    if (!res.ok || !res.body) throw new Error(`llama.cpp error: ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let lineEnd: number;
      while ((lineEnd = buffer.indexOf('\n')) !== -1) {
        let line = buffer.slice(0, lineEnd).trim();
        buffer = buffer.slice(lineEnd + 1);
        if (!line) continue;
        if (line.startsWith('data:')) line = line.slice(5).trim();
        if (line === '[DONE]') return;
        const obj = safeJsonParse(line);
        if (!obj) continue;

        if (obj.choices?.length) {
          const chunk =
              obj.choices[0].text ?? obj.choices[0].delta?.content ?? '';
          if (chunk) yield chunk as string;
          continue;
        }
        if (typeof obj.content === 'string') {
          yield obj.content;
          if (obj.stop) return;
        }
      }
    }
  }

  // OpenAI-compatible chat streaming
  private async *
      streamFromOpenAI(
          base: string, model: string, prompt: string, apiKey: string,
          maxTokens: number) {
    const url = `${base.replace(/\/$/, '')}/v1/chat/completions`;
    const body = {
      model,
      stream: true,
      max_tokens: Math.max(1, Math.min(maxTokens, 256)),
      messages: [{role: 'user', content: prompt}],
      temperature: 0.2,
    };
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) throw new Error(`OpenAI error: ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let idx: number;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        let line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        if (line.startsWith('data:')) line = line.slice(5).trim();
        if (line === '[DONE]') return;

        const obj = safeJsonParse(line);
        if (!obj) continue;
        const delta = obj.choices?.[0]?.delta?.content;
        if (typeof delta === 'string') yield delta;
      }
    }
  }
}

/* ------------------------------------------------------------------------------------------------
 * Ghost Text state (unchanged semantics)
 * ----------------------------------------------------------------------------------------------*/
const setGhost = StateEffect.define<string|null>();
const ghostField = StateField.define<string|null>({
  create() {
    return null;
  },
  update(value, tr) {
    for (const e of tr.effects) {
      if (e.is(setGhost)) return e.value;
    }
    if (tr.docChanged || tr.selection) return null;  // clear on edit/move
    return value;
  },
});

function ghostDecorations(view: EditorView): DecorationSet {
  const text = view.state.field(ghostField, false);
  const builder = new RangeSetBuilder<Decoration>();
  if (!text) return builder.finish();

  const sel = view.state.selection.main;
  if (!sel.empty) return builder.finish();

  const deco = Decoration.widget({widget: new GhostWidget(text), side: 1});
  builder.add(sel.from, sel.from, deco);
  return builder.finish();
}

class GhostWidget extends WidgetType {
  constructor(readonly text: string) {
    super();
  }
  toDOM() {
    const span = document.createElement('span');
    span.textContent = this.text;
    span.style.opacity = '0.5';
    span.style.fontStyle = 'italic';
    span.style.pointerEvents = 'none';
    return span;
  }
}

// Keymap: Ctrl-Space accepts ghost text; Escape clears
const acceptGhost = () => {
  return (view: EditorView) => {
    const text = view.state.field(ghostField, false);
    if (!text) return false;
    view.dispatch({
      changes: [{from: view.state.selection.main.from, insert: text + ' '}],
      selection: {anchor: view.state.selection.main.from + text.length + 1},
      effects: setGhost.of(null),
    });
    return true;
  };
};

const ghostKeymap = keymap.of([
  {key: 'Ctrl-Space', preventDefault: true, run: acceptGhost()},
  {
    key: 'Escape',
    run: (view) => {
      if (view.state.field(ghostField, false)) {
        view.dispatch({effects: setGhost.of(null)});
        return true;
      }
      return false;
    },
  },
]);

/* ------------------------------------------------------------------------------------------------
 * ViewPlugin: Prediction (streaming ghost text) — preserves your original logic
 *  - Adds AbortController to prevent illegal CM6 cycle updates and race
 * conditions
 * ----------------------------------------------------------------------------------------------*/
function createGhostPlugin(
    predictor: StreamPredictor, getSettings: () => ScribePilotSettings) {
  let aborter: AbortController|null = null;
  let debounceTimer: number|null = null;
  let lastContext = '';

  return ViewPlugin.fromClass(class {
    decorations: DecorationSet;
    extensions = [ghostField, ghostKeymap];

    constructor(readonly view: EditorView) {
      this.decorations = Decoration.none;
      this.schedule();
    }

    update(update: ViewUpdate) {
      // Trigger on doc changes; you can also trigger on cursor moves if
      // desired.
      if (update.docChanged) this.schedule();
    }

    destroy() {
      this.cancel();
    }

    private schedule() {
      const {debounceMs} = getSettings();
      if (debounceTimer) window.clearTimeout(debounceTimer);
      debounceTimer =
          window.setTimeout(() => this.request(), Math.max(50, debounceMs));
    }

    private clearGhost() {
      this.view.dispatch({effects: setGhost.of(null)});
    }

    private cancel() {
      if (aborter) {
        aborter.abort();
        aborter = null;
      }
      if (debounceTimer) {
        window.clearTimeout(debounceTimer);
        debounceTimer = null;
      }
    }

    private getPrompt(): string|null {
      const sel = this.view.state.selection.main;
      if (!sel.empty) return null;

      // Take a limited context window before cursor
      const ctxLen = 1200;
      const from = Math.max(0, sel.from - ctxLen);
      const prefix = this.view.state.doc.sliceString(from, sel.from);

      // Ignore if last run’s context is identical or whitespace-only
      if (!prefix.trim()) {
        this.clearGhost();
        return null;
      }
      return prefix;
    }

    private async request() {
      const prompt = this.getPrompt();
      if (!prompt) return;
      if (prompt === lastContext) return;
      lastContext = prompt;

      this.cancel();  // cancel previous
      aborter = new AbortController();

      try {
        let full = '';
        const it = predictor.predictStream(prompt);
        let result = await it.next();
        while (!result.done) {
          if (aborter.signal.aborted) throw new Error('aborted');
          full += (result.value ?? '');
          // Keep it short and non-invasive; stop at first newline
          const display = (full.split('\n')[0] || '').slice(0, 120);
          this.view.dispatch({effects: setGhost.of(display)});
          result = await it.next();
        }
      } catch (e) {
        // Optional debug: console.warn('Prediction error', e);
      } finally {
        aborter = null;
      }
    }
  }, {
    provide: (plugin) =>
        [ghostKeymap, ghostField,
         EditorView.decorations.compute([ghostField], () => ghostDecorations)],
  });
}

/* ------------------------------------------------------------------------------------------------
 * ViewPlugin: SymSpell typo underline (lightweight, local)
 *  - Underlines the current/last word when likely misspelled
 *  - Offers a command to apply the top suggestion
 * ----------------------------------------------------------------------------------------------*/
const typoMarksField = StateField.define<DecorationSet>({
  create() {
    return Decoration.none;
  },
  update(value, tr) {
    if (!tr.docChanged && !tr.selection) return value;
    // cleared by plugin when recomputed
    return Decoration.none;
  }
});

function createTypoPlugin(
    sym: SymSpellLite, getSettings: () => ScribePilotSettings) {
  const underlineDeco = (from: number, to: number) =>
      Decoration
          .mark({
            class: 'scribepilot-typo-underline',
            attributes: {'aria-label': 'Possible spelling error'}
          })
          .range(from, to);

  return ViewPlugin.fromClass(class {
    decorations: DecorationSet = Decoration.none;
    private debounce: number|null = null;

    constructor(private view: EditorView) {
      this.queue();
    }
    update(u: ViewUpdate) {
      if (u.docChanged || u.selectionSet) this.queue();
    }

    private queue() {
      const s = getSettings();
      if (!s.enableSymSpell || !s.underlineTypos) {
        this.clear();
        return;
      }

      if (this.debounce) window.clearTimeout(this.debounce);
      this.debounce = window.setTimeout(() => this.recompute(), 140);
    }

    private clear() {
      this.decorations = Decoration.none;
      this.view.dispatch({effects: StateEffect.appendConfig.of([])});
    }

    private recompute() {
      const s = getSettings();
      if (!s.enableSymSpell || !s.underlineTypos) {
        this.clear();
        return;
      }

      const doc = this.view.state.doc;
      const sel = this.view.state.selection.main;
      const ctxLen = 100;
      const from = Math.max(0, sel.from - ctxLen);
      const prefix = doc.sliceString(from, sel.from);

      // Extract last token (simple word chars)
      const m = prefix.match(/([A-Za-z']+)$/);
      if (!m) {
        this.decorations = Decoration.none;
        this.view.dispatch({effects: StateEffect.appendConfig.of([])});
        return;
      }

      const word = m[1];
      if (word.length < 3) {
        this.decorations = Decoration.none;
        this.view.dispatch({effects: StateEffect.appendConfig.of([])});
        return;
      }

      const suggestion = sym.suggest(word);
      if (!suggestion) {
        this.decorations = Decoration.none;
        this.view.dispatch({effects: StateEffect.appendConfig.of([])});
        return;
      }

      const start = sel.from - word.length;
      const deco = underlineDeco(start, sel.from);
      const rs = new RangeSetBuilder<Decoration>();
      rs.add(start, sel.from, deco.value);
      this.decorations = rs.finish();
    }
  }, {
    decorations: v => v.decorations,
    provide: plugin => [typoMarksField],
  });
}

/* ------------------------------------------------------------------------------------------------
 * LanguageTool integration (grammar/style)
 *  - Debounced check of the last sentence (or small context) to limit requests
 *  - Highlights offending ranges; command to apply first suggestion at cursor
 * ----------------------------------------------------------------------------------------------*/
const ltMarksField = StateField.define<DecorationSet>({
  create() {
    return Decoration.none;
  },
  update(value, tr) {
    if (tr.docChanged || tr.selection) {
      // Clear on edits; plugin recomputes with debounce
      return Decoration.none;
    }
    return value;
  }
});

function createLanguageToolPlugin(getSettings: () => ScribePilotSettings) {
  const grammarDeco = (from: number, to: number) =>
      Decoration
          .mark({
            class: 'scribepilot-grammar-underline',
            attributes: {'aria-label': 'Possible grammar/style issue'}
          })
          .range(from, to);

  let debounceTimer: number|null = null;

  return ViewPlugin.fromClass(class {
    decorations: DecorationSet = Decoration.none;

    constructor(private view: EditorView) {
      this.schedule();
    }
    update(u: ViewUpdate) {
      if (u.docChanged || u.selectionSet) this.schedule();
    }

    private schedule() {
      const s = getSettings();
      if (!s.enableLanguageTool) {
        this.decorations = Decoration.none;
        return;
      }
      if (debounceTimer) window.clearTimeout(debounceTimer);
      debounceTimer =
          window.setTimeout(() => this.check(), Math.max(200, s.ltDebounceMs));
    }

    private async check() {
      const s = getSettings();
      if (!s.enableLanguageTool) {
        this.decorations = Decoration.none;
        return;
      }

      const view = this.view;
      const sel = view.state.selection.main;
      const ctxStart = Math.max(0, sel.from - 800);
      const context = view.state.doc.sliceString(ctxStart, sel.from);
      // Only check after a sentence boundary or on meaningful text
      if (!context.trim()) {
        this.decorations = Decoration.none;
        return;
      }
      if (!isPunctuationBoundary(context.slice(-1))) {
        // Still allow if long clause
        if (context.length < 60) {
          this.decorations = Decoration.none;
          return;
        }
      }

      try {
        const params = new URLSearchParams({
          text: lastSentenceFragment(context, 500),
          language: s.ltLanguage || 'en-US',
        });
        const res = await fetch(`${s.ltEndpoint.replace(/\/$/, '')}/v2/check`, {
          method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: params,
        });
        if (!res.ok) return;

        const data = await res.json();
        const matches: Array<{
          offset: number,
          length: number,
          message: string,
          shortMessage: string,
          replacements: Array<{value: string}>
        }> = data.matches ?? [];

        // Convert offsets to doc positions
        const sentence = params.get('text') || '';
        const sentenceStart = sel.from - sentence.length;
        const builder = new RangeSetBuilder<Decoration>();
        for (const m of matches) {
          const from = Math.max(0, sentenceStart + m.offset);
          const to =
              Math.max(from, Math.min(view.state.doc.length, from + m.length));
          builder.add(from, to, grammarDeco(from, to).value);
        }
        this.decorations = builder.finish();
      } catch {
        // ignore errors silently
      }
    }
  }, {
    decorations: v => v.decorations,
    provide: plugin => [ltMarksField],
  });
}

/* ------------------------------------------------------------------------------------------------
 * Plugin main
 * ----------------------------------------------------------------------------------------------*/
export default class ScribePilotPlugin extends Plugin {
  settings: ScribePilotSettings;
  private predictor!: StreamPredictor;
  public symspell = new SymSpellLite();

  async onload() {
    await this.loadSettings();

    // Init SymSpell
    try {
      if (this.settings.enableSymSpell) {
        if (this.settings.symSpellDictUrl) {
          await this.symspell.loadFromUrl(
              this.settings.symSpellDictUrl, this.settings.symSpellDistance);
        } else {
          this.symspell.loadTinySeed(this.settings.symSpellDistance);
        }
      }
    } catch (e) {
      console.warn('SymSpell dictionary load failed:', e);
      new Notice(
          'ScribePilot: Failed to load SymSpell dictionary (see console).');
    }

    // Predictor
    this.predictor = new StreamPredictor(() => this.settings);

    // Editor extensions
    this.registerEditorExtension(
        createGhostPlugin(this.predictor, () => this.settings));

    if (this.settings.enableSymSpell) {
      this.registerEditorExtension(
          createTypoPlugin(this.symspell, () => this.settings));
      // Provide minimal CSS for underline styles
      this.injectStyles();
    }

    if (this.settings.enableLanguageTool) {
      this.registerEditorExtension(
          createLanguageToolPlugin(() => this.settings));
      this.injectStyles();  // shared CSS
    }

    // Commands
    this.addCommand({
      id: 'scribepilot-insert-sample',
      name: 'Insert sample prediction (debug)',
      editorCallback: (editor: Editor) => {
        editor.replaceRange(' [Predicted text]', editor.getCursor());
      },
    });

    this.addCommand({
      id: 'scribepilot-apply-spell-suggestion',
      name: 'Apply top spelling suggestion (cursor word)',
      editorCallback: (editor: Editor) => {
        if (!this.settings.enableSymSpell) return;
        const cursor = editor.getCursor();
        const line = editor.getLine(cursor.line);
        const left = line.slice(0, cursor.ch);
        const m = left.match(/([A-Za-z']+)$/);
        if (!m) return;
        const word = m[1];
        const sug = this.symspell.suggest(word);
        if (!sug) {
          new Notice('No spelling suggestion.');
          return;
        }
        const start = cursor.ch - word.length;
        editor.replaceRange(
            sug.suggestion, {line: cursor.line, ch: start},
            {line: cursor.line, ch: cursor.ch});
      },
    });

    this.addCommand({
      id: 'scribepilot-apply-first-grammar-fix',
      name: 'Apply first LanguageTool suggestion (near cursor)',
      checkCallback: (checking) => {
        if (!this.settings.enableLanguageTool) return false;
        const md = this.app.workspace.getActiveViewOfType(MarkdownView);
        if (!md) return false;
        if (checking) return true;
        this.applyNearestLtFix(md.editor);
        return true;
      }
    });

    this.addSettingTab(new ScribePilotSettingTab(this.app, this));
    new Notice('ScribePilot loaded (Ctrl-Space to accept predictions).');
  }

  onunload() {}

  private async applyNearestLtFix(editor: Editor) {
    // Lightweight re-check of the sentence around cursor, then apply first
    // replacement
    if (!this.settings.enableLanguageTool) return;
    const cursor = editor.getCursor();
    const ctxStart = {line: Math.max(0, cursor.line - 6), ch: 0};
    const ctxEnd = {line: cursor.line, ch: cursor.ch};
    const context = editor.getRange(ctxStart, ctxEnd);
    const text = lastSentenceFragment(context, 400);
    if (!text.trim()) return;

    const params = new URLSearchParams(
        {text, language: this.settings.ltLanguage || 'en-US'});
    try {
      const res = await fetch(
          `${this.settings.ltEndpoint.replace(/\/$/, '')}/v2/check`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: params,
          });
      if (!res.ok) return;
      const data = await res.json();
      const matches: Array<{
        offset: number,
        length: number,
        replacements: Array<{value: string}>
      }> = data.matches ?? [];
      if (!matches.length || !matches[0].replacements?.length) {
        new Notice('No grammar suggestion found.');
        return;
      }
      const sentenceStartPos = editor.posToOffset(ctxEnd) - text.length;
      const m0 = matches[0];
      const fromOff = sentenceStartPos + m0.offset;
      const toOff = fromOff + m0.length;
      const from = editor.offsetToPos(fromOff);
      const to = editor.offsetToPos(toOff);
      editor.replaceRange(m0.replacements[0].value, from, to);
    } catch {
      // ignore
    }
  }

  private injectStyles() {
    const id = 'scribepilot-styles';
    if (document.getElementById(id)) return;
    const style = document.createElement('style');
    style.id = id;
    style.textContent = `
      .cm-content .scribepilot-typo-underline {
        text-decoration: underline wavy red;
        text-underline-offset: 2px;
      }
      .cm-content .scribepilot-grammar-underline {
        text-decoration: underline dotted orange;
        text-underline-offset: 2px;
      }
    `;
    document.head.appendChild(style);
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }
  async saveSettings() {
    await this.saveData(this.settings);
  }
}

/* ------------------------------------------------------------------------------------------------
 * Settings UI — extends your original with provider=OpenAI, cloud mode,
 * SymSpell & LanguageTool
 * ----------------------------------------------------------------------------------------------*/
class ScribePilotSettingTab extends PluginSettingTab {
  constructor(app: App, private plugin: ScribePilotPlugin) {
    super(app, plugin);
  }

  display(): void {
    const {containerEl} = this;
    containerEl.empty();
    containerEl.createEl('h2', {text: 'ScribePilot Settings'});

    // Mode
    new Setting(containerEl)
        .setName('Mode')
        .setDesc('Offline (local), Hybrid (local→cloud), or Cloud only')
        .addDropdown(
            d => d.addOptions({
                    offline: 'Offline',
                    hybrid: 'Hybrid',
                    cloud: 'Cloud',
                  })
                     .setValue(this.plugin.settings.mode)
                     .onChange(async (v) => {
                       this.plugin.settings.mode =
                           v as ScribePilotSettings['mode'];
                       await this.plugin.saveSettings();
                     }));

    // Provider
    new Setting(containerEl)
        .setName('Provider')
        .setDesc(
            'Auto, Ollama (local), llama.cpp (local), or OpenAI-compatible (cloud)')
        .addDropdown(
            d => d.addOptions({
                    auto: 'Auto',
                    ollama: 'Ollama',
                    llamacpp: 'llama.cpp',
                    openai: 'OpenAI-compatible',
                  })
                     .setValue(this.plugin.settings.provider)
                     .onChange(async (v) => {
                       this.plugin.settings.provider =
                           v as ScribePilotSettings['provider'];
                       await this.plugin.saveSettings();
                     }));

    // Endpoint
    new Setting(containerEl)
        .setName('Endpoint')
        .setDesc(
            'Leave empty for defaults (Ollama :11434, llama.cpp :8080, OpenAI https://api.openai.com)')
        .addText(
            t => t.setPlaceholder('http://localhost:11434')
                     .setValue(this.plugin.settings.endpoint)
                     .onChange(async (v) => {
                       this.plugin.settings.endpoint = v.trim();
                       await this.plugin.saveSettings();
                     }));

    // Model
    new Setting(containerEl)
        .setName('Model')
        .setDesc(
            'Ollama (e.g., llama3:8b), llama.cpp GGUF name, or OpenAI model (e.g., gpt-4o-mini)')
        .addText(
            t => t.setPlaceholder('llama3:8b')
                     .setValue(this.plugin.settings.model)
                     .onChange(async (v) => {
                       this.plugin.settings.model = v.trim();
                       await this.plugin.saveSettings();
                     }));

    // API Key (for cloud)
    new Setting(containerEl)
        .setName('API Key (Cloud)')
        .setDesc(
            'Required for cloud mode (OpenAI-compatible). Stored locally in this vault.')
        .addText(
            t => t.setPlaceholder('sk-...')
                     .setValue(this.plugin.settings.apiKey)
                     .onChange(async (v) => {
                       this.plugin.settings.apiKey = v.trim();
                       await this.plugin.saveSettings();
                     }));

    // Max tokens
    new Setting(containerEl)
        .setName('Max tokens')
        .setDesc('Maximum tokens to predict per request')
        .addText((t) => {
          t.inputEl.type = 'number';
          t.setValue(String(this.plugin.settings.maxTokens))
              .onChange(async (v: string) => {
                const n = Math.max(1, Math.min(256, Number(v) || 32));
                this.plugin.settings.maxTokens = n;
                await this.plugin.saveSettings();
              });
        });

    // Debounce
    new Setting(containerEl)
        .setName('Debounce (ms)')
        .setDesc('Delay before requesting a prediction')
        .addText((t) => {
          t.inputEl.type = 'number';
          t.setValue(String(this.plugin.settings.debounceMs))
              .onChange(async (v: string) => {
                const n = Math.max(50, Math.min(1000, Number(v) || 180));
                this.plugin.settings.debounceMs = n;
                await this.plugin.saveSettings();
              });
        });

    containerEl.createEl('h3', {text: 'Spelling (SymSpell)'});
    new Setting(containerEl)
        .setName('Enable SymSpell')
        .addToggle(
            t => t.setValue(this.plugin.settings.enableSymSpell)
                     .onChange(async (v) => {
                       this.plugin.settings.enableSymSpell = v;
                       await this.plugin.saveSettings();
                     }));

    new Setting(containerEl)
        .setName('Edit distance')
        .setDesc('1 = stricter, 2 = more tolerant (slower)')
        .addDropdown(
            d => d.addOptions({'1': '1', '2': '2'})
                     .setValue(String(this.plugin.settings.symSpellDistance))
                     .onChange(async (v) => {
                       this.plugin.settings.symSpellDistance =
                           (v === '1' ? 1 : 2);
                       await this.plugin.saveSettings();
                     }));

    new Setting(containerEl)
        .setName('Dictionary URL')
        .setDesc(
            'Optional: URL to a frequency word list ("word frequency" lines). Leave blank to use tiny built-in seed.')
        .addText(
            t => t.setPlaceholder('https://.../frequency_dictionary_en.txt')
                     .setValue(this.plugin.settings.symSpellDictUrl)
                     .onChange(async (v) => {
                       this.plugin.settings.symSpellDictUrl = v.trim();
                       await this.plugin.saveSettings();
                     }));

    new Setting(containerEl)
        .setName('Underline typos')
        .addToggle(
            t => t.setValue(this.plugin.settings.underlineTypos)
                     .onChange(async (v) => {
                       this.plugin.settings.underlineTypos = v;
                       await this.plugin.saveSettings();
                     }));

    containerEl.createEl('h3', {text: 'LanguageTool (Grammar/Style)'});
    new Setting(containerEl)
        .setName('Enable LanguageTool')
        .addToggle(
            t => t.setValue(this.plugin.settings.enableLanguageTool)
                     .onChange(async (v) => {
                       this.plugin.settings.enableLanguageTool = v;
                       await this.plugin.saveSettings();
                     }));

    new Setting(containerEl)
        .setName('Endpoint')
        .setDesc(
            'Public: https://api.languagetool.org (rate-limited). You can self-host too.')
        .addText(
            t => t.setPlaceholder('https://api.languagetool.org')
                     .setValue(this.plugin.settings.ltEndpoint)
                     .onChange(async (v) => {
                       this.plugin.settings.ltEndpoint = v.trim();
                       await this.plugin.saveSettings();
                     }));

    new Setting(containerEl)
        .setName('Language')
        .addText(
            t => t.setPlaceholder('en-US')
                     .setValue(this.plugin.settings.ltLanguage)
                     .onChange(async (v) => {
                       this.plugin.settings.ltLanguage = v.trim() || 'en-US';
                       await this.plugin.saveSettings();
                     }));

    new Setting(containerEl)
        .setName('Debounce (ms)')
        .setDesc('Delay before running a grammar check on the last sentence')
        .addText(t => {
          t.inputEl.type = 'number';
          t.setValue(String(this.plugin.settings.ltDebounceMs))
              .onChange(async (v) => {
                this.plugin.settings.ltDebounceMs =
                    Math.max(200, Math.min(2000, Number(v) || 700));
                await this.plugin.saveSettings();
              });
        });

    // Helper buttons
    new Setting(containerEl)
        .setName('Reload SymSpell dictionary')
        .setDesc('Reload the dictionary from URL or built-in seed.')
        .addButton(btn => btn.setButtonText('Reload').onClick(async () => {
          try {
            if (this.plugin.settings.symSpellDictUrl) {
              await this.plugin.symspell.loadFromUrl(
                  this.plugin.settings.symSpellDictUrl,
                  this.plugin.settings.symSpellDistance);
            } else {
              this.plugin.symspell.loadTinySeed(
                  this.plugin.settings.symSpellDistance);
            }
            new Notice('SymSpell dictionary reloaded.');
          } catch (e) {
            console.error(e);
            new Notice('Failed to reload SymSpell dictionary.');
          }
        }));
  }
}
