# free-model-aggregator-01

A scripted router for the make.fun `free-model-aggregator-01` bounty: one
long-running Node 22 server (TypeScript, stdlib only, zero dependencies) that
registers with the trial env and answers its 20 requests with the best free
model that meets each request's requirements. No model calls of its own — the
answers come from the ten simulated providers, through the env.

## Layout

- `main.ts` — the entrypoint: reads the brief, probes catalogs, registers,
  serves `POST /v1/chat/completions`, `POST /v1/messages` and
  `GET /v1/models`, polls `GET /work` and exits 0 when the work is done.
- `lib/brief.ts` — the brief's three JSON documents (provider directory,
  sealed quality table, workload manifest), the catalog probe, and the
  catalog document `--list` prints and `/register` posts.
- `lib/wire.ts` — the client wires (OpenAI, Anthropic) and the three
  upstream wires (OpenAI, Anthropic, Gemini), and the translations between
  them, in JSON and in server-sent events.
- `lib/router.ts` — provider selection, failover, and the per-call retry.

## How it works

**Startup.** The brief at `$MAKEFUN_PROMPT_FILE` carries three JSON
documents, found by their `schema` member. Each provider's catalog route is
probed once, in its own wire (`GET <base>/v1/models`, or
`/v1beta/models` for Gemini; `anthropic-version: 2023-06-01` on the
Anthropic wire). The catalog is built from the probes: one entry a provider
— id, `base_url` (the directory's own path, `/p/<id>`, never joined to
`$MAKEFUN_ENV_URL`), `key_env`, wire, and models, each with id, free and
`context_length`. The free flag comes from the quality table's row where it
has one, else the listing's own word. The same document is what `--list`
prints and what `/register` posts, byte for byte; with no brief and no env
set, `--list` prints `{"providers":[]}` and exits 0. A failed probe leaves
its provider out of the catalog and it gets no traffic.

**Routing.** For each request, the candidates are the free models that meet
its requirements (tools, vision, JSON, minimum context), on the wire its
JSON mode allows — an OpenAI JSON-mode request never goes to an
Anthropic-wire provider, whose wire cannot carry JSON mode — ordered by
rating, then by the brief's provider and model order. The client sends the
requests one at a time, in manifest order; `GET /work` names the request
being sent, with a counter as the fallback. Calls to one provider are
sequential — the providers charge a 429 for a concurrent call.

**Wires.** The request is translated into the provider's wire without
rewriting anything the client compares: the system text, the last user
message's text, the tool names in order, `max_tokens`, the stop sequences,
the image bytes, and JSON mode. The answer comes back in the wire the
client used, streamed or not as it asked, with the provider's text whole,
its usage counts, and its stop reason mapped (`end_turn` to `stop`,
`max_tokens` to `length`, `tool_use` to `tool_calls`, `refusal` to
`content_filter`). Streams are rebuilt in the client's wire from the
provider's frames, in the right order: OpenAI chunks end with one finish
chunk, a usage chunk, then `[DONE]`; Anthropic events run `message_start`,
content blocks, `message_delta`, `message_stop`. A streamed OpenAI answer
always carries a usage chunk — the request asks for one
(`stream_options.include_usage`), and a same-wire stream that went through
untouched gets one appended if it lacks it. Usage is the provider's own
formula, computed from the client's request and the answer text.

**Failover.** A provider's failure schedule is one or two single call
numbers, so a 402/429 is not, on its own, the end of a provider: the same
model is retried once (the next call is a new call number and usually
passes), and only three refusals in a row prove the budget spent or an
outage begun — a provider has at most two scheduled failures — and only
then is it set aside for the rest of the trial. A 5xx or network error is
retried on the next candidate; any other non-200 is not. Exactly one
provider call answers each request.

**Determinism.** Every route is a function of the brief, the providers'
answers and the requests seen so far: no wall clock, no unseeded
randomness, fixed timestamps in the model listings. The catalog's
`base_url` is the provider's own path, so a replay on a fresh env origin
produces the same document.

## Local test results

Docker is not installed on this VM, so `grade-local --driver docker` cannot
run. The kit's real env (`program/env/`) and scorer (`program/scorer/`)
were run directly instead: the env on a loopback port, the entry with
`MAKEFUN_PROMPT_FILE` and `MAKEFUN_ENV_URL` set, then SIGTERM to the env
for the state export, then the scorer. All three practice trials:

| trial | valid | upstream calls | score (quality / served / wire / failover / catalog) |
|-------|-------|----------------|------------------------------------------------------|
| 0 | 20 of 20 | 33 | **99** (99 / 100 / 100 / 100 / 100) |
| 1 | 20 of 20 | 33 | **99** (99 / 100 / 100 / 100 / 100) |
| 2 | 20 of 20 | 35 | **99** (99 / 100 / 100 / 100 / 100) |

All 233 checks passed on every trial; both `GET /v1/models` routes listed
exactly the free models. Quality 99 is the ceiling on these trials: the
best model (1475) has a budget of 12 calls with two scheduled failures, so
at most 10 of the 20 requests can be answered at 1475 — the entry serves
the 10 highest-weight ones at 1475 and the rest at 1470 (quality numerator
38,213 of 38,248 on trial-0, the optimum).

Also verified separately:

- `node main.ts --list` with no brief prints `{"providers":[]}`, exit 0.
- `node main.ts --list` against the trial-0 env prints a catalog whose
  free/paid model sets, `base_url`, `key_env` and wire match
  `practice/trial-0/reference/catalog.json` for all ten providers.
- `npx tsc --strict` (with `noUncheckedIndexedAccess`, `noUnusedLocals`)
  passes clean.
- Two bugs found and fixed during testing: a Gemini SSE stream was
  misread as JSON (the stream flag now comes from the request, not the
  translated body), and a provider was blacklisted on its first 429 (now
  three refusals in a row, plus one retry per model). Before the fixes
  trial-0 scored 88 with 18 of 20 valid.

## Running it

```sh
node main.ts --list        # print the catalog, exit 0
node main.ts [port]        # serve on 0.0.0.0:<port> (ephemeral when omitted)
```

With `$MAKEFUN_PROMPT_FILE` and `$MAKEFUN_ENV_URL` set it probes, registers,
serves the 20 requests, and exits 0 when `GET /work` says done. Without
them it serves as a standalone router with no providers.
