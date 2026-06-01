# Planner Empty `tool_calls` Investigation

Date: 2026-06-01

## Scope

This note records the investigation for the reproduced planner failures in the WebUI private chat prompt range:

- `规划器/WebUI用户的私聊/1780064622647`
- through `规划器/WebUI用户的私聊/1780064664560`

The reported symptom was that the planner kept failing because no executable `tool_calls` were returned.

## Relevant Artifacts

Prompt HTML files:

- `logs/maisaka_prompt/planner/webui_private_webui_user_webui_k3h4m0d36_mp3w48a3/1780064622647.html`
- through `logs/maisaka_prompt/planner/webui_private_webui_user_webui_k3h4m0d36_mp3w48a3/1780064664560.html`

Matching planner debug cache files:

- `logs/debug_planner_cache/20260529_222342_645255_planner_9f90cdb2d96d016cbdd3ac7f89fe1388_qwen3.6-27b.json`
- through `logs/debug_planner_cache/20260529_222424_559505_planner_9f90cdb2d96d016cbdd3ac7f89fe1388_qwen3.6-27b.json`

The timestamp range maps to local time `2026-05-29 22:23:42` through `22:24:24`.

## Observations

There are 11 planner requests in this range:

1. The initial planner request.
2. Ten empty-`tool_calls` correction retries.

All 11 requests had:

- `response_body.tool_calls=[]`
- `final_response_body.tool_calls=[]`
- upstream raw `ChatCompletionMessage(... tool_calls=[], content=...)`
- `finish_reason='stop'`, not `tool_calls`

The model repeatedly wrote tool-like text into `content` instead of returning OpenAI-compatible structured `message.tool_calls`.

Observed raw content formats included:

- `<tool_search query="f1_standings"/>`
- fenced JSON containing `"tool_calls"`
- `<tool_calls_method><method_name>web_search</method_name>...`
- `{"name": "web_search", "arguments": {...}}`
- `<tool_calls><tool_call><tool_name>tool_search</tool_name>...`

These formats are not standard OpenAI-compatible `tool_calls`, and they are not covered by the current MaiBot XML fallback parser.

## Native OpenAI SDK vs MaiBot Request Comparison

The debug cache's serialized prompt payload was compared with the actual `provider_request.request_kwargs` captured from MaiBot's `OpenaiClient`.

Findings:

- `messages` in the planner cache matched `provider_request.request_kwargs.messages` exactly.
- `tool_definitions` matched `provider_request.request_kwargs.tools` exactly.
- After restoring `max_tokens=28000` from config because the persisted debug payload redacted it, the native OpenAI SDK kwargs and MaiBot `OpenaiClient` kwargs were equivalent:
  - `DIFF_KEYS []`
  - `VALUES_EQUAL True`

The request shape was:

- `model='qwen3.6-27b'`
- `temperature=0.6`
- `max_tokens=28000`
- `stream=False`
- `tools=9`
- `messages=4` initially, `messages=5` during correction retries
- `extra_body={'top_p': 0.95, 'top_k': 20, 'min_p': 0, 'presence_penalty': 0, 'repetition_penalty': 1}`

A live native SDK replay could not be completed because the upstream endpoint was unreachable during investigation:

- `http://fgs6.bakamoe.com:9092/v1/models` failed with `ConnectError: All connection attempts failed`
- `http://fgs6.bakamoe.com:9092/health` failed with `ConnectError: All connection attempts failed`

## Likely Root Cause

The failure is most likely a Qwen3.6 + vLLM tool parser mismatch/instability triggered by this F1 standings prompt, not a MaiBot request-conversion issue.

The upstream OpenAI-compatible response itself contained `tool_calls=[]`. MaiBot did not drop structured tool calls after receiving them; there were no structured calls to drop.

This aligns with vLLM/Qwen tool-calling behavior:

- With `tool_choice='auto'`, vLLM lets the model generate freely, then a selected tool parser extracts calls from raw text.
- If the model emits a format the parser does not recognize, tool-like text remains in `content` and `message.tool_calls` stays empty.
- Qwen reasoning/thinking modes have known failure modes where tool calls are emitted as XML/JSON-ish text rather than structured OpenAI `tool_calls`.

Historical cache comparison showed that the same `qwen3.6-27b` model can return structured `tool_calls` successfully in other planner turns, so this is not a universal request-shape failure.

## Additional Findings

- The current empty-`tool_calls` retry behavior worked as designed: it retried ten times and did not append bad planner responses to chat history.
- The correction prompt was present from retry 1/10 through 10/10, but it did not force the model/parser into a structured `tool_calls` response.
- The debug provider-request sanitizer currently redacts keys containing `token`, which turns `max_tokens` into `<redacted>` in persisted debug payloads. This does not affect live requests, but it makes replay less convenient and should be adjusted.

## Recommended Next Steps

1. Test planner with `qwen3.6-27b-nothink`, or add `extra_body.chat_template_kwargs.enable_thinking=false` for the planner's qwen model.
2. If planner must rely on this model/parser combination, add a MaiBot fallback parser for the concrete malformed formats observed above.
3. Adjust provider-request debug sanitization so `max_tokens` is not redacted.
4. When the upstream endpoint is reachable again, replay the same captured request with native `AsyncOpenAI.chat.completions.create(**kwargs)` using these variants:
   - exact MaiBot kwargs
   - exact kwargs plus `tool_choice='auto'`
   - exact kwargs plus `tool_choice='required'`
   - exact kwargs without `extra_body`
   - exact kwargs with `extra_body.chat_template_kwargs.enable_thinking=false`
5. Prefer a server-side fix if available: verify the vLLM launch flags for Qwen3.6, especially `--enable-auto-tool-choice`, `--tool-call-parser`, reasoning parser, and default chat template kwargs.
