# Decision protocol fixtures

These are offline examples adapted to the common test rubric. They are not
recordings from authenticated calls. The OpenAI answer order is intentionally
reversed to test association by name rather than position.

- `typesafe.json`: the [System One response](https://docs.typesafe.ai/api/system-one)
  envelope; choice and score use the Jev distribution format. Omits optional
  routing metadata and exercises a zero predicate probability.
- `openrouter.json`: the [OpenRouter decisions example](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request),
  with routing ID, provider, cost, and the published choice/score probabilities.
- `openai.json`: [SDK 3.57.0 Decision types](https://github.com/openai/openai-python/blob/v3.57.0/src/openai/types/decision.py),
  with numeric score indices and detailed token usage.

The codecs accept these examples without network access. A separate rounding
test checks synthetic three-decimal distributions against RAGU's 0.01 sum
tolerance. Live opt-in tests check service compatibility separately.
