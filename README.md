# ScribePilot – AI-Assisted Writing for Obsidian

ScribePilot adds **inline text prediction**, **autocorrect**, and **grammar suggestions** to Obsidian, with privacy-first, offline-by-default design.

## Features

-   **Inline text prediction** – Ghost text appears as you type; press `Tab` to accept.
-   **Autocorrect & spellcheck** – Instant typo correction with quick-fix menu.
-   **Grammar & style suggestions** – Underlines issues with explanations and fixes.
-   **Local-first AI** – Runs with local models (Ollama/Llama.cpp, LanguageTool) for zero cost and full privacy.
-   **Cloud fallback (optional)** – Use OpenAI or LanguageTool Premium if preferred.
-   **Vault personalization** – Learns your vocabulary and style from your notes.

## Development Roadmap

**Phase 1**

-   Repo scaffolding (npm).
-   Settings UI skeleton; provider interfaces.
-   CM6 extension for ghost text & simple suggestions.

**Phase 2**

-   **SymSpell** integration + user dictionary.
-   **LanguageTool** provider (local + public endpoints).
-   Diagnostics rendering and quick-fix command.

**Phase 3**

-   **Ollama/Llama.cpp** prediction provider (streaming).
-   Ranking/fusion logic; debounce, max-token caps.
-   Status bar mode switch; per-vault config.

**Phase 4**

-   **Personalization index**: build n-gram counts from vault (job queue, throttle, ignore code blocks).
-   Mix n-gram priors into ranking.

**Phase 5**

-   Performance profiling; cache; cancellation; edge cases (mobile, reading view).
-   Telemetry (opt-in, anonymous) for latency/error only.

**Phase 6**

-   Tests (unit + integration with mock providers).
-   Docs (README, setup guides for Ollama/Llama.cpp + LanguageTool).
-   Security & privacy review; defaults to Offline.

**Phase 7–8** (public beta)

-   Community feedback cycle, bug fixes.
-   Submit to **Obsidian Community Plugins**.

---
