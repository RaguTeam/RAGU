"""Mock HTTP server for testing OpenAI-compatible API clients (CachedAsyncOpenAI)."""
from __future__ import annotations

import http.server
import json
import random
import time
import threading
from typing import Any, cast

# JSON-compatible value: the only types the mock server needs to produce.
JsonValue = str | int | float | bool | list[Any] | dict[str, Any] | None


# ---------------------------------------------------------------------------
# Part 1: sample-data construction from a JSON Schema dict (server-side)
# ---------------------------------------------------------------------------

def make_sample_from_json_schema(
    schema: dict[str, Any],
    defs: dict[str, Any] | None = None,
) -> JsonValue:
    """Return a minimal JSON-compatible sample value from a JSON Schema dict.

    The server receives a JSON Schema (serialised by the OpenAI SDK from
    response_format) and must reply with a conforming JSON string — it never
    sees Python types.  Handles: primitives, $ref/$defs, anyOf/oneOf,
    object (properties), array (items).
    """
    defs = defs if defs is not None else cast(dict[str, Any], schema.get('$defs', {}))

    # Resolve $ref — e.g. "#/$defs/MyModel" → look up in defs
    if '$ref' in schema:
        def_name = schema['$ref'].split('/')[-1]
        return make_sample_from_json_schema(cast(dict[str, Any], defs[def_name]), defs)

    # anyOf / oneOf — pick first non-null branch
    for key in ('anyOf', 'oneOf'):
        if key in schema:
            non_null = [o for o in schema[key] if o.get('type') != 'null']
            return make_sample_from_json_schema(non_null[0], defs) if non_null else None

    json_type = schema.get('type')
    if json_type == 'string':
        return 'str'
    if json_type == 'integer':
        return 0
    if json_type == 'number':
        return 0.0
    if json_type == 'boolean':
        return False
    if json_type == 'null':
        return None
    if json_type == 'array':
        items = schema.get('items')
        return [make_sample_from_json_schema(cast(dict[str, Any], items), defs)] if items else []
    if json_type == 'object':
        return {
            name: make_sample_from_json_schema(cast(dict[str, Any], prop), defs)
            for name, prop in schema.get('properties', {}).items()
        }
    return None


# ---------------------------------------------------------------------------
# Part 2: mock HTTP server
# ---------------------------------------------------------------------------

class _OpenAIMockHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler that routes incoming POST requests to OpenAIMockServer methods."""

    server: OpenAIMockServer  # type: ignore[assignment]

    def do_POST(self) -> None:
        """Fault-check → sleep(random delay) → read body → route → write JSON response (or '.' if broken schema)."""
        srv = self.server
        active, exception_code, broken_schema = srv.fault_snapshot()

        if not active:
            self.send_response(503)
            self.send_header('Connection', 'close')
            self.end_headers()
            return

        if exception_code is not None:
            self.send_response(exception_code)
            self.send_header('Connection', 'close')
            self.end_headers()
            return

        if srv.check_rate_limit():
            self.send_response(429)
            self.send_header('Connection', 'close')
            self.end_headers()
            return

        lo, hi = srv.default_delay
        if hi > 0.0:
            time.sleep(random.uniform(lo, hi))

        length = int(self.headers.get('Content-Length', 0))
        body = cast(dict[str, Any], json.loads(self.rfile.read(length)))

        result = srv.route(self.path, body)

        response_bytes = b'.' if broken_schema else json.dumps(result).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default per-request log output to keep test output clean."""
        pass


class OpenAIMockServer(http.server.ThreadingHTTPServer):
    """Thread-backed mock HTTP server that returns minimal valid OpenAI-compatible responses.

    Intended for use with CachedAsyncOpenAI(base_url=server.base_url, api_key='mock').
    Handles: POST /v1/chat/completions, POST /v1/embeddings, POST /v1/score.
    """

    RequestHandlerClass = _OpenAIMockHandler
    _thread: threading.Thread
    _lock: threading.Lock
    _timers: list[threading.Timer]   # all pending timers; cancelled on stop()
    default_delay: tuple[float, float]
    min_delay: float                 # minimum seconds between requests; excess → 429
    _last_request_time: float        # monotonic timestamp of the last accepted request

    # Fault-injection state — written from Timer threads, read from the server
    # thread; all access must be protected by _lock.
    _active: bool                    # False → 503 + Connection: close
    _exception_code: int | None      # not None → return this HTTP error code instead of 200
    _broken_schema: bool             # True → return "." instead of proper JSON body

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 0,
        default_delay: tuple[float, float] = (0.0, 0.0),
        min_delay: float = 0.0,
    ) -> None:
        """Bind socket, initialise fault-injection state, store default_delay range for do_POST sleep."""
        super().__init__((host, port), _OpenAIMockHandler)
        self.default_delay = default_delay
        self.min_delay = min_delay
        self._last_request_time = 0.0
        self._lock = threading.Lock()
        self._active = True
        self._exception_code = None
        self._broken_schema = False
        self._timers = []

    @property
    def base_url(self) -> str:
        """Return 'http://host:port/v1/' ready to pass as AsyncOpenAI(base_url=...)."""
        host, port = cast(tuple[str, int], self.server_address)
        return f'http://{host}:{port}/v1/'

    def start(self) -> None:
        """Spawn a background daemon thread running serve_forever()."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Cancel all pending timers, call shutdown(), and join the background thread."""
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
        self.shutdown()
        self._thread.join()

    def fault_snapshot(self) -> tuple[bool, int | None, bool]:
        """Return (active, exception_code, broken_schema) atomically; called by do_POST."""
        with self._lock:
            return self._active, self._exception_code, self._broken_schema

    def set_min_delay(self, min_delay: float) -> None:
        """Update min_delay and reset _last_request_time so the next request is always accepted."""
        with self._lock:
            self.min_delay = min_delay
            self._last_request_time = 0.0

    def check_rate_limit(self) -> bool:
        """Return True (→ 429) if elapsed since last accepted request < min_delay, else stamp and return False."""
        with self._lock:
            now = time.monotonic()
            if self.min_delay > 0.0 and now - self._last_request_time < self.min_delay:
                return True
            self._last_request_time = now
            return False

    def route(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Dispatch path to the matching _handle_* method; called by do_POST."""
        if path.endswith('/chat/completions'):
            return self._handle_chat_completions(body)
        if path.endswith('/embeddings'):
            return self._handle_embeddings(body)
        if path.endswith('/score'):
            return self._handle_score(body)
        return {}

    # -- active / inactive -------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Flip _active; thread and socket are untouched — only do_POST behaviour changes."""
        with self._lock:
            self._active = active

    def schedule_inactive(self, start_time: float, inactive_time: float) -> None:
        """Non-blocking: set_active(False) after start_time s, set_active(True) after start_time+inactive_time s."""
        t1 = threading.Timer(start_time, self.set_active, [False])
        t2 = threading.Timer(start_time + inactive_time, self.set_active, [True])
        t1.daemon = True
        t2.daemon = True
        self._timers += [t1, t2]
        t1.start()
        t2.start()

    # -- HTTP error code injection -----------------------------------------

    def set_exception(self, active: bool, err_code: int) -> None:
        """If active, do_POST replies err_code+Connection:close instead of 200; if not active, clears _exception_code."""
        with self._lock:
            self._exception_code = err_code if active else None

    def schedule_exception(self, start_time: float, inactive_time: float, err_code: int) -> None:
        """Non-blocking: set_exception(True, err_code) after start_time s, set_exception(False, err_code) after start_time+inactive_time s."""
        t1 = threading.Timer(start_time, self.set_exception, [True, err_code])
        t2 = threading.Timer(start_time + inactive_time, self.set_exception, [False, err_code])
        t1.daemon = True
        t2.daemon = True
        self._timers += [t1, t2]
        t1.start()
        t2.start()

    # -- broken-schema injection -------------------------------------------

    def set_broken_schema(self, active: bool) -> None:
        """If active, do_POST writes '.' as the response body instead of a valid JSON payload."""
        with self._lock:
            self._broken_schema = active

    def schedule_broken_schema(self, start_time: float, inactive_time: float) -> None:
        """Non-blocking: set_broken_schema(True) after start_time s, set_broken_schema(False) after start_time+inactive_time s."""
        t1 = threading.Timer(start_time, self.set_broken_schema, [True])
        t2 = threading.Timer(start_time + inactive_time, self.set_broken_schema, [False])
        t1.daemon = True
        t2.daemon = True
        self._timers += [t1, t2]
        t1.start()
        t2.start()

    # -- endpoint handlers -------------------------------------------------

    def _handle_chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return a ChatCompletion dict: 'str' content for plain text, JSON-encoded schema sample
        for response_format requests, or a tool_call message for tool-mode requests."""
        tools = cast(list[dict[str, Any]], body['tools']) if body.get('tools') else None
        response_format = cast(dict[str, Any], body['response_format']) if body.get('response_format') else None

        if tools:
            # as_tool=True path: generate sample from function parameters schema
            fn = cast(dict[str, Any], tools[0]['function'])
            arguments_json = json.dumps(
                make_sample_from_json_schema(cast(dict[str, Any], fn['parameters']))
            )
            message: dict[str, Any] = {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{
                    'id': 'call_mock',
                    'type': 'function',
                    'function': {'name': cast(str, fn['name']), 'arguments': arguments_json},
                }],
            }
            finish_reason = 'tool_calls'
        elif response_format and response_format.get('type') == 'json_schema':
            # beta parse path: extract schema from response_format.json_schema.schema
            schema = cast(dict[str, Any],
                cast(dict[str, Any], response_format['json_schema'])['schema'])
            message = {
                'role': 'assistant',
                'content': json.dumps(make_sample_from_json_schema(schema)),
            }
            finish_reason = 'stop'
        else:
            # plain text path
            message = {'role': 'assistant', 'content': 'str'}
            finish_reason = 'stop'

        return {
            'id': 'chatcmpl-mock',
            'object': 'chat.completion',
            'created': 0,
            'model': cast(str, body.get('model', 'mock')),
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        }

    def _handle_embeddings(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return a CreateEmbeddingResponse dict.

        Supports both single-text (``input="text"``) and batch
        (``input=["t1", "t2", ...]``) requests.  Each input item
        produces one all-zero embedding vector in the response ``data``
        list, preserving the original order.
        """
        raw_input = body.get('input', '')
        n = len(raw_input) if isinstance(raw_input, list) else 1
        return {
            'object': 'list',
            'data': [
                {'object': 'embedding', 'embedding': [0.0], 'index': i}
                for i in range(n)
            ],
            'model': cast(str, body.get('model', 'mock')),
            'usage': {'prompt_tokens': 0, 'total_tokens': 0},
        }

    def _handle_score(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return a score-endpoint response with one zero-score entry per item in body['text_2']."""
        text_2 = cast(list[str], body.get('text_2', []))
        return {'data': [{'index': i, 'score': 0.0} for i in range(len(text_2))]}


if __name__ == '__main__':
    import asyncio
    from pydantic import BaseModel
    from ragu.models.openai import CachedAsyncOpenAI

    class Answer(BaseModel):
        reasoning: str
        score: int

    async def _main() -> None:
        server = OpenAIMockServer()
        server.start()
        try:
            client = CachedAsyncOpenAI(base_url=server.base_url, api_key='mock')

            embedding = await client.embed_text(model_name='mock', text='hello')
            print('embedding :', embedding)

            scores = await client.score(
                model_name='mock', text_1='hello', text_2=['world', 'foo'],
            )
            print('score     :', scores)

            text = await client.chat_completion(
                model_name='mock',
                conversation=[{'role': 'user', 'content': 'Say something.'}],
                output_schema=str,
            )
            print('str       :', text)

            answer = await client.chat_completion(
                model_name='mock',
                conversation=[{'role': 'user', 'content': 'Rate this.'}],
                output_schema=Answer,
            )
            print('BaseModel :', answer)
        finally:
            server.stop()

    asyncio.run(_main())
