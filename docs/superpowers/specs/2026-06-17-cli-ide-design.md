# LatticeMemory CLI IDE Design

## Goal

Build a first public-release slice of a CLI-based IDE as `lattice ide`: a terminal command center that lets a developer configure BYOK AI access, operate LatticeMemory features, inspect local project state, and bridge into VS Code without requiring a full VS Code extension in the first iteration.

## Scope

This first slice is a keyboard-driven command shell, not a graphical TUI and not a full IDE replacement. It should feel like a practical local operator console:

- Start with `lattice ide`.
- Store local BYOK provider settings without committing secrets.
- Run AI chat through OpenAI-compatible endpoints.
- Route Q&A through `LatticeQABot` when a local Q&A cache is loaded.
- Expose existing LatticeMemory operations: cache inspect/export/import, proxy start guidance, analytics, dedup, gaps, drift, vertical smoke actions, and agent-memory inspection.
- Bridge to VS Code through the installed `code` command for opening files/workspaces and listing/installing extensions.

Excluded from this slice:

- Building a VS Code extension package.
- Editing source files through an AI agent.
- Autonomous multi-file coding.
- Full-screen terminal layout with panes.
- Direct support for every proprietary AI API shape. Non-OpenAI providers can work through OpenAI-compatible endpoints.

## User Experience

`lattice ide` opens an interactive prompt:

```text
LatticeMemory IDE
workspace: E:\latticememory
provider: unset

lm> help
lm> provider set openai --base-url https://api.openai.com/v1 --model gpt-4o-mini
lm> chat "Explain this cache hit rate"
lm> proxy analytics --port 8000
lm> cache inspect --cache helpdesk.db
lm> vscode open README.md
lm> exit
```

The shell must also accept one-shot commands:

```bash
lattice ide chat "What can this repo do?"
lattice ide cache inspect --cache helpdesk.db
lattice ide vscode open README.md
```

One-shot commands make the feature testable and scriptable. Interactive mode can be thin command dispatch around the same handlers.

## Architecture

Add a small `latticememory.ide` package with focused modules:

- `latticememory/ide/cli.py`: `argparse` parser, interactive loop, and command dispatch.
- `latticememory/ide/config.py`: local config load/save, environment override handling, and secret redaction.
- `latticememory/ide/providers.py`: OpenAI-compatible BYOK client using `urllib.request` so no new dependency is required.
- `latticememory/ide/workspace.py`: workspace detection, file search helpers, and safe path resolution.
- `latticememory/ide/vscode.py`: VS Code CLI bridge using `code` when available.
- `latticememory/ide/lattice_ops.py`: thin wrappers over existing CLI functions and library APIs.

The existing `latticememory.cli` remains the top-level entry point. It gains an `ide` subcommand that imports `latticememory.ide.cli` lazily.

## BYOK Provider Model

Provider config lives in a user-local JSON file:

```text
Windows: %APPDATA%\latticememory\ide_config.json
macOS/Linux: ~/.config/latticememory/ide_config.json
```

Environment variables override config:

- `LATTICE_IDE_BASE_URL`
- `LATTICE_IDE_API_KEY`
- `LATTICE_IDE_MODEL`

The config file may store a base URL and model. API keys should be written only when the user explicitly passes `--save-key`; otherwise keys come from environment variables or the current process. Any command that prints provider config must redact keys.

The first provider implementation targets OpenAI-compatible chat completions:

```http
POST {base_url}/chat/completions
Authorization: Bearer {api_key}
```

This covers OpenAI, many local gateways, LiteLLM, OpenRouter-style gateways, and self-hosted compatible servers. Provider-specific SDKs are out of scope for the first slice.

## LatticeMemory Feature Coverage

The IDE should expose existing functionality rather than reimplement it:

- `cache inspect/export/import`: call existing CLI handlers or shared helper functions.
- `proxy analytics`: call `/v1/analytics`.
- `proxy doctor`: check `/health`, `/openapi.json`, admin-key-gated cache access when a key is supplied.
- `dedup`: call existing dedup command handler.
- `gaps` and `drift`: call `LatticeFlywheel` through existing CLI behavior.
- `qa load` and `qa ask`: wrap `LatticeQABot` for local Q&A workflows.
- `verticals list`: show shipped vertical classes and their primary methods.
- `agent keys`: inspect `AgentMemorySync`-style exported key manifests when available.

If a function needs a heavy encoder download, the command must say so before loading it. The command should fail with a clear message if optional dependencies are missing.

## VS Code Bridge

The first version integrates with VS Code through the `code` command only:

- `vscode status`: detect whether `code` is available and print version.
- `vscode open [path]`: open a workspace, folder, or file.
- `vscode extensions list`: run `code --list-extensions`.
- `vscode extensions install <id>`: run `code --install-extension <id>`.

The bridge must not assume a VS Code extension exists. A real extension can be a later product slice once the command shell proves useful.

## Error Handling

Commands should return nonzero exit codes in one-shot mode and continue the loop in interactive mode. Errors should include:

- Missing file or workspace path.
- Missing provider base URL, API key, or model.
- Provider HTTP error with status and short body.
- Missing optional dependency.
- Missing VS Code CLI.
- Proxy unreachable.

Secrets must not be printed in errors or logs.

## Testing

Unit tests should cover:

- Config load/save and environment override precedence.
- Secret redaction.
- Provider request body and auth header construction using a fake local HTTP server or monkeypatched opener.
- One-shot command parsing for `chat`, `provider`, `cache`, and `vscode`.
- VS Code bridge behavior with monkeypatched subprocess calls.
- Interactive dispatch of a single command without requiring a real terminal.

Integration smoke tests should cover:

- `python -m latticememory.ide.cli --help`.
- `python -m latticememory.cli ide --help`.
- `lattice ide provider show` with no configured key.

Browser testing is not required for this slice because the product surface is terminal-based. Existing proxy and extension browser checks remain separate release checks.

## Release Criteria

The first slice is ready when:

- `lattice ide --help` and `lattice ide` both work.
- A user can configure an OpenAI-compatible BYOK endpoint and send a chat message.
- A user can run at least one cache command, one proxy diagnostic, one VS Code bridge command, and one LatticeMemory feature command from the IDE shell.
- Existing tests plus new IDE tests pass.
- README documents the new `lattice ide` quickstart, BYOK environment variables, and VS Code bridge limitations.

## Follow-On Slices

After this shell is stable:

1. Add a `textual` full-screen TUI with panes for chat, cache, logs, and command output.
2. Add a real VS Code extension that talks to the local IDE/proxy over HTTP.
3. Add AI-assisted code editing with explicit diff review, tests, and no autonomous writes by default.
4. Add provider adapters for Anthropic, Gemini, Ollama, and Azure OpenAI when they cannot be reached through an OpenAI-compatible gateway.
