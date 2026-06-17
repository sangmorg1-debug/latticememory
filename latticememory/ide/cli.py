from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .config import IdeConfig, load_config, provider_from_env, save_config
from .lattice_ops import list_verticals, proxy_analytics, proxy_doctor
from .providers import ProviderError, chat_completion
from .vscode import VSCodeUnavailable, install_extension, list_extensions, open_path, status
from .workspace import resolve_workspace_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lattice ide", description="LatticeMemory terminal IDE")
    sub = parser.add_subparsers(dest="command")

    provider = sub.add_parser("provider", help="Configure BYOK provider")
    provider_sub = provider.add_subparsers(dest="provider_command", required=True)
    provider_sub.add_parser("show", help="Show redacted provider config")
    set_p = provider_sub.add_parser("set", help="Set provider config")
    set_p.add_argument("provider", choices=["openai"], help="Provider compatibility profile")
    set_p.add_argument("--base-url", required=True)
    set_p.add_argument("--model", required=True)
    set_p.add_argument("--api-key", default="")
    set_p.add_argument("--save-key", action="store_true")

    chat = sub.add_parser("chat", help="Send a BYOK AI chat message")
    chat.add_argument("prompt")

    proxy = sub.add_parser("proxy", help="Proxy diagnostics")
    proxy_sub = proxy.add_subparsers(dest="proxy_command", required=True)
    doctor = proxy_sub.add_parser("doctor")
    doctor.add_argument("--host", default="127.0.0.1")
    doctor.add_argument("--port", type=int, default=8000)
    doctor.add_argument("--admin-key", default=None)
    analytics = proxy_sub.add_parser("analytics")
    analytics.add_argument("--host", default="127.0.0.1")
    analytics.add_argument("--port", type=int, default=8000)

    verticals = sub.add_parser("verticals", help="Vertical tools")
    verticals_sub = verticals.add_subparsers(dest="verticals_command", required=True)
    verticals_sub.add_parser("list")

    vscode = sub.add_parser("vscode", help="VS Code bridge")
    vscode_sub = vscode.add_subparsers(dest="vscode_command", required=True)
    vscode_sub.add_parser("status")
    open_p = vscode_sub.add_parser("open")
    open_p.add_argument("path")
    ext = vscode_sub.add_parser("extensions")
    ext_sub = ext.add_subparsers(dest="extensions_command", required=True)
    ext_sub.add_parser("list")
    install = ext_sub.add_parser("install")
    install.add_argument("extension_id")

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return interactive_loop()
    return dispatch(build_parser().parse_args(argv))


def interactive_loop() -> int:
    print("LatticeMemory IDE")
    print(f"workspace: {Path.cwd()}")
    print("type 'help' for commands, 'exit' to quit")
    while True:
        try:
            line = input("lm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"exit", "quit"}:
            return 0
        if line == "help":
            build_parser().print_help()
            continue
        try:
            code = main(shlex.split(line))
        except SystemExit:
            code = 2
        if code:
            print(f"command failed: {code}")


def dispatch(args: argparse.Namespace) -> int:
    try:
        if args.command == "provider":
            return _provider(args)
        if args.command == "chat":
            cfg = provider_from_env(load_config())
            print(chat_completion(cfg, args.prompt))
            return 0
        if args.command == "proxy":
            if args.proxy_command == "doctor":
                data = proxy_doctor(host=args.host, port=args.port, admin_key=args.admin_key)
            else:
                data = proxy_analytics(host=args.host, port=args.port)
            print(json.dumps(data, indent=2))
            return 0
        if args.command == "verticals":
            for row in list_verticals():
                print(f"{row['class']}: {row['capability']}")
            return 0
        if args.command == "vscode":
            return _vscode(args)
    except (ProviderError, VSCodeUnavailable, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    build_parser().print_help()
    return 0


def _provider(args: argparse.Namespace) -> int:
    if args.provider_command == "show":
        print(json.dumps(provider_from_env(load_config()).redacted(), indent=2))
        return 0
    api_key = args.api_key if args.save_key else ""
    path = save_config(IdeConfig(base_url=args.base_url, model=args.model, api_key=api_key))
    print(f"Saved provider config: {path}")
    if args.api_key and not args.save_key:
        print("API key was not saved. Use LATTICE_IDE_API_KEY or pass --save-key.")
    return 0


def _vscode(args: argparse.Namespace) -> int:
    if args.vscode_command == "status":
        print(json.dumps(status(), indent=2))
    elif args.vscode_command == "open":
        open_path(resolve_workspace_path(Path.cwd(), args.path))
    elif args.vscode_command == "extensions" and args.extensions_command == "list":
        for extension in list_extensions():
            print(extension)
    elif args.vscode_command == "extensions" and args.extensions_command == "install":
        install_extension(args.extension_id)
        print(f"Installed {args.extension_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
