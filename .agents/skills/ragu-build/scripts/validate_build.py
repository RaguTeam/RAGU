#!/usr/bin/env python3
"""Validate a generated RAGU build script without making any network calls.

Catches the dominant failure mode of generated code: invented classes, wrong import
paths, non-existent constructor parameters and typos in ``Settings``.

Three levels:
  1. syntax and module import;
  2. a static check of every RAGU class call against the real ``__init__`` signature
     (nothing is executed);
  3. an actual run of the script's ``main()`` from a temp directory, with every call
     that would reach a model server replaced by a local stand-in — so settings,
     storage backends, the graph and the engine get constructed for real, and nothing
     leaves the process.

Usage:
    python .claude/skills/ragu-build/scripts/validate_build.py build_my.py
    python .claude/skills/ragu-build/scripts/validate_build.py build_my.py --no-runtime
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import importlib
import importlib.util
import inspect
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

Problem = Tuple[str, str]  # (severity, message): "ERROR" | "WARN"


# --------------------------------------------------------------------------- #
# Level 2: static signature checking
# --------------------------------------------------------------------------- #

def _collect_imports(tree: ast.Module) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Collect the names imported from ``ragu``.

    :returns: (symbols, modules), where symbols maps a local name to
        "module:attribute" and modules maps a local module alias to its full name.
    """
    symbols: Dict[str, str] = {}
    modules: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] != "ragu" or node.level:
                continue
            for alias in node.names:
                symbols[alias.asname or alias.name] = f"{node.module}:{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "ragu":
                    continue
                modules[alias.asname or alias.name] = alias.name

    return symbols, modules


def _dotted(node: ast.AST) -> str | None:
    """Flatten ``a.b.c`` into a string; return None for anything else."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _resolve(
    name: str,
    symbols: Dict[str, str],
    modules: Dict[str, str],
) -> Tuple[Any, str] | None:
    """Find the object a call refers to, if it comes from ragu."""
    if name in symbols:
        module_name, attr = symbols[name].split(":")
    elif "." in name:
        head, _, attr = name.rpartition(".")
        module_name = modules.get(head)
        if module_name is None:
            # ragu.models.llm.LLMOpenAI reached through a plain `import ragu`
            if head.split(".")[0] in modules or head.split(".")[0] == "ragu":
                module_name = head
            else:
                return None
    else:
        return None

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None

    target = getattr(module, attr, None)
    if target is None:
        return None
    return target, f"{module_name}.{attr}"


def _check_call(target: Any, qualname: str, call: ast.Call) -> List[Problem]:
    problems: List[Problem] = []

    if not (inspect.isclass(target) or inspect.isfunction(target)):
        return problems

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return problems

    params = signature.parameters
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    accepts_var_pos = any(p.kind is p.VAR_POSITIONAL for p in params.values())
    positional_slots = [
        p for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    line = call.lineno

    for keyword in call.keywords:
        if keyword.arg is None:  # **kwargs at the call site — nothing to check
            continue
        if keyword.arg not in params and not accepts_var_kw:
            known = ", ".join(sorted(params)[:12])
            problems.append((
                "ERROR",
                f"line {line}: {qualname}(...) takes no parameter "
                f"'{keyword.arg}'. Available: {known}",
            ))

    given_positional = sum(1 for a in call.args if not isinstance(a, ast.Starred))
    has_star = any(isinstance(a, ast.Starred) for a in call.args)
    if not accepts_var_pos and not has_star and given_positional > len(positional_slots):
        problems.append((
            "ERROR",
            f"line {line}: {qualname}(...) takes at most "
            f"{len(positional_slots)} positional arguments, {given_positional} given",
        ))

    if not has_star:
        supplied = {p.name for p in positional_slots[:given_positional]}
        supplied |= {k.arg for k in call.keywords if k.arg}
        for param in params.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is param.empty and param.name not in supplied:
                if any(k.arg is None for k in call.keywords):
                    continue
                problems.append((
                    "ERROR",
                    f"line {line}: {qualname}(...) is missing required "
                    f"parameter '{param.name}'",
                ))

    return problems


def _check_settings(tree: ast.Module, symbols: Dict[str, str]) -> List[Problem]:
    """Check that every attribute assigned on Settings actually exists."""
    aliases = {
        name for name, origin in symbols.items()
        if origin.endswith(":Settings")
    }
    if not aliases:
        return []

    try:
        from ragu.common.global_parameters import Settings
    except Exception:
        return []

    problems: List[Problem] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in aliases
                and not hasattr(Settings, target.attr)
            ):
                problems.append((
                    "ERROR",
                    f"line {node.lineno}: Settings has no attribute '{target.attr}'",
                ))
    return problems


def static_checks(path: Path) -> List[Problem]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [("ERROR", f"syntax error at line {exc.lineno}: {exc.msg}")]

    symbols, modules = _collect_imports(tree)
    problems: List[Problem] = []

    for name, origin in symbols.items():
        module_name, attr = origin.split(":")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            problems.append(("ERROR", f"cannot import module {module_name}: {exc}"))
            continue
        if not hasattr(module, attr):
            problems.append(("ERROR", f"{module_name} has no name '{attr}'"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name is None:
            continue
        resolved = _resolve(name, symbols, modules)
        if resolved is None:
            continue
        target, qualname = resolved
        problems.extend(_check_call(target, qualname, node))

    problems.extend(_check_settings(tree, symbols))
    return problems


# --------------------------------------------------------------------------- #
# Level 3: run the script with every network call neutralized
# --------------------------------------------------------------------------- #

EMBEDDING_DIM = 1536


def _env_names(tree: ast.Module) -> List[str]:
    """Collect the environment variables the script reads.

    Covers ``os.environ["X"]``, ``os.environ.get("X")`` and ``os.getenv("X")`` — the
    validator fills in whatever is missing so a build can be checked without a
    configured environment.
    """
    names: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _dotted(node.value) == "os.environ":
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.append(key.value)
        elif isinstance(node, ast.Call) and node.args:
            call = _dotted(node.func)
            if call in ("os.getenv", "os.environ.get"):
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.append(first.value)

    return names


def _placeholder(name: str) -> str:
    """A value that survives client construction but reaches nothing."""
    if name.endswith("_URL") or "BASE_URL" in name:
        return "http://localhost:1/v1"
    return "validation-stub"


def _descendants(cls: type) -> List[type]:
    found = [cls]
    for subclass in cls.__subclasses__():
        found.extend(_descendants(subclass))
    return found


def _override(cls: type, name: str, value: Any) -> None:
    """Replace a method on a class and on every subclass that redefines it."""
    for klass in _descendants(cls):
        if klass is cls or name in klass.__dict__:
            setattr(klass, name, value)


def _neutralize() -> None:
    """Replace everything that would talk to a model server with a local stand-in.

    What remains is the real assembly path: settings, storage backends, the graph
    and the engine are constructed exactly as the script constructs them.
    """
    from ragu.graph.knowledge_graph import KnowledgeGraph
    from ragu.models.embedder import EmbedderOpenAI
    from ragu.search_engine.base_engine import BaseEngine
    from ragu.utils.token_truncation import TokenTruncation

    async def anoop(*args: Any, **kwargs: Any) -> None:
        return None

    async def answer(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(query=query, response="<validation stub>", metrics={})

    async def answers(self: Any, queries: List[str], *args: Any, **kwargs: Any) -> Any:
        return [SimpleNamespace(query=q, response="<validation stub>", metrics={}) for q in queries]

    EmbedderOpenAI.initialize = anoop
    EmbedderOpenAI.dim = property(lambda self: EMBEDDING_DIM)
    EmbedderOpenAI.embed_text = anoop
    EmbedderOpenAI.batch_embed_text = anoop

    # A "local" tokenizer backend pulls the tokenizer from HuggingFace inside the
    # constructor, which is a network call like any other.
    TokenTruncation.__init__ = lambda self, *args, **kwargs: None
    TokenTruncation.__call__ = lambda self, text, return_stats=False: text

    # Same for a local reranker: constructing it downloads the weights unless they
    # happen to be cached. Optional dependency, so only patched when installed.
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        pass
    else:
        CrossEncoder.__init__ = lambda self, *args, **kwargs: None

    KnowledgeGraph.build_from_docs = anoop

    _override(BaseEngine, "query", answer)
    _override(BaseEngine, "search", answer)
    _override(BaseEngine, "batch_query", answers)
    _override(BaseEngine, "batch_search", answers)


def runtime_checks(path: Path) -> List[Problem]:
    _neutralize()

    # Only ever set what is absent, so a configured environment is left untouched.
    # Nothing here reaches a server: every call that would leave the process is
    # replaced by _neutralize() above.
    for name in _env_names(ast.parse(path.read_text(encoding="utf-8"))):
        os.environ.setdefault(name, _placeholder(name))

    spec = importlib.util.spec_from_file_location("_ragu_build_under_test", path)
    if spec is None or spec.loader is None:
        return [("ERROR", f"cannot load {path}")]
    module = importlib.util.module_from_spec(spec)

    # Register before executing: dataclasses resolve postponed annotations through
    # sys.modules, and a module missing from it breaks @dataclass at class creation.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return [("ERROR", "module does not import:\n" + traceback.format_exc(limit=6))]

    entry = getattr(module, "main", None)
    if not inspect.iscoroutinefunction(entry):
        return [("WARN", "no async main() found; ran import checks only")]

    takes_flag = bool(inspect.signature(entry).parameters)

    # Run from a temp directory so relative storage paths land there, not in the
    # user's workspace.
    origin = Path.cwd()
    os.chdir(tempfile.mkdtemp(prefix="ragu-validate-"))
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            asyncio.run(entry(False) if takes_flag else entry())
    except Exception:
        return [("ERROR", "main() raised:\n" + traceback.format_exc(limit=8))]
    finally:
        os.chdir(origin)

    return []


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="The generated build_<name>.py")
    parser.add_argument("--no-runtime", action="store_true",
                        help="Static checks only; do not construct any objects.")
    args = parser.parse_args()

    if not args.script.is_file():
        print(f"File not found: {args.script}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(args.script.resolve().parent))

    problems = static_checks(args.script)
    if not args.no_runtime and not any(sev == "ERROR" for sev, _ in problems):
        problems.extend(runtime_checks(args.script))

    errors = [m for sev, m in problems if sev == "ERROR"]
    warnings = [m for sev, m in problems if sev == "WARN"]

    for message in warnings:
        print(f"WARN  {message}")
    for message in errors:
        print(f"ERROR {message}")

    if errors:
        print(f"\nFailed: {len(errors)} error(s), {len(warnings)} warning(s).")
        return 1

    print(f"OK: {args.script} assembles. Warnings: {len(warnings)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
