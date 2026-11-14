// The entry: one long-running server that registers with the trial env and
// answers its 20 requests, then exits 0 when the env says the work is done.
//
// Startup: read the brief from $MAKEFUN_PROMPT_FILE, probe each provider's
// catalog route, build the catalog (the same document `--list` prints),
// register it with POST $MAKEFUN_ENV_URL/register, and serve. A process
// still running when the trial's wall is spent scores 0, so the server polls
// GET $MAKEFUN_ENV_URL/work and exits 0 as soon as it says done.
//
// A request is answered by the best free model that meets its requirements:
// the manifest entry for it (the client sends the requests one at a time, in
// manifest order; GET /work names the request being sent, and a counter is
// the fallback), the candidates from the router, one provider call at a
// time, the next candidate on a 402/429. The answer comes back in the wire
// the client used, streamed or not as it asked.
//
// Run with --list: print the catalog as one JSON document and exit 0
// without starting the server. With no brief and no env set, that document
// is {"providers":[]}. Run with --help: print usage and exit 0.
//
// Determinism: every route is a function of the brief, the providers'
// answers and the requests seen so far — no wall clock, no unseeded
// randomness. The only time-based values are the poll and retry intervals,
// which never reach a response.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  collectCatalog,
  freeModelIds,
  isRecord,
  parseBrief,
  probeProvider,
  type Brief,
  type CatalogDocument,
  type ClientRequest,
  type Json,
  type ProviderCatalog,
} from './lib/brief.ts';
import { Router, translateJsonAnswer, translateSseAnswer } from './lib/router.ts';
import {
  anthropicError,
  anthropicModelList,
  ensureOpenAIUsageChunk,
  openAIError,
  openAIModelList,
  upstreamStreamText,
  usageOf,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const MAX_BODY_BYTES = 2 * 1024 * 1024;
const WORK_POLL_MS = 250;
const REGISTER_RETRY_MS = 2000;
const WORK_LOOKUP_TIMEOUT_MS = 500;

/** The request's requirements as the body itself shows them, when there is no manifest to read. */
function requirementsFromBody(api: 'openai' | 'anthropic', body: Json): Json {
  const requirements: Json = {};
  if (api === 'openai') {
    if (Array.isArray(body.tools) && body.tools.length > 0) requirements.tools = true;
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of messages) {
      if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
      for (const part of raw.content) {
        if (isRecord(part) && part.type === 'image_url') requirements.vision = true;
      }
    }
    if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
      requirements.json = true;
    }
  } else {
    if (Array.isArray(body.tools) && body.tools.length > 0) requirements.tools = true;
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of messages) {
      if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
      for (const block of raw.content) {
        if (isRecord(block) && block.type === 'image') requirements.vision = true;
      }
    }
  }
  return requirements;
}

interface ServerState {
  brief: Brief;
  envUrl: string;
  catalog: CatalogDocument;
  router: Router;
  served: ServedModel[];
  /** The manifest index of the request expected next; the counter fallback. */
  nextIndex: number;
  registered: boolean;
  exiting: boolean;
}

/** The manifest request the client is sending now: GET /work names it while it is in flight. */
async function currentRequest(state: ServerState): Promise<ClientRequest | null> {
  const manifest = state.brief.requests;
  if (manifest.length === 0) return null;
  if (state.envUrl !== '') {
    try {
      const response = await fetch(`${state.envUrl.replace(/\/+$/, '')}/work`, {
        signal: AbortSignal.timeout(WORK_LOOKUP_TIMEOUT_MS),
      });
      if (response.ok) {
        const work: unknown = await response.json();
        if (isRecord(work) && typeof work.next === 'string') {
          const index = manifest.findIndex((r) => r.id === work.next);
          if (index >= 0) {
            state.nextIndex = index + 1;
            const found = manifest[index];
            if (found !== undefined) return found;
          }
        }
      }
    } catch {
      // The env did not answer in time: fall through to the counter.
    }
  }
  const index = Math.min(state.nextIndex, manifest.length - 1);
  state.nextIndex = index + 1;
  const found = manifest[index];
  return found ?? null;
}

function sendJson(res: ServerResponse, status: number, value: Json): void {
  const body = JSON.stringify(value);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) });
  res.end(body);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  res.end(text);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('request body is over the cap'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

/** One request, answered with the best free model that meets its requirements. */
async function handleCompletion(
  state: ServerState,
  api: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<void> {
  const stream = body.stream === true;
  const jsonMode =
    api === 'openai' &&
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const manifest = await currentRequest(state);
  const request: ClientRequest =
    manifest ?? {
      id: 'standalone',
      api,
      stream,
      weight: 1,
      requirements: requirementsFromBody(api, body),
    };
  const candidates = state.router.candidatesFor(request, jsonMode);
  for (const candidate of candidates) {
    let result;
    try {
      result = await state.router.call(candidate, api, body);
    } catch {
      continue;
    }
    if (result.status !== 200) {
      if (!result.retriable) break;
      continue;
    }
    if (stream) {
      const answerText = upstreamStreamText(candidate.wire, result.text);
      const usage = usageOf(body, api, answerText);
      if (candidate.wire === 'openai' && api === 'openai') {
        // Same wire: the provider's stream goes through untouched, with a
        // usage chunk guaranteed (the request asked for one; this is the backstop).
        sendSse(res, ensureOpenAIUsageChunk(result.text, candidate.modelId, usage));
      } else {
        sendSse(res, translateSseAnswer(candidate, api, result.text, usage));
      }
      return;
    }
    if (result.body === null) continue;
    sendJson(res, 200, translateJsonAnswer(candidate, api, result.body));
    return;
  }
  // No candidate answered: the request is lost, but the trial goes on.
  const message = 'no free model that meets this request answered';
  if (api === 'openai') sendJson(res, 503, openAIError(message, 'server_error', null));
  else sendJson(res, 503, anthropicError('api_error', message));
}

async function handleRequest(state: ServerState, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const path = url.pathname;
  if (req.method === 'GET' && path === '/v1/models') {
    const models =
      typeof req.headers['x-api-key'] === 'string' && req.headers['x-api-key'] !== ''
        ? anthropicModelList(state.served)
        : openAIModelList(state.served);
    sendJson(res, 200, models);
    return;
  }
  if (req.method === 'GET' && (path === '/' || path === '/health')) {
    sendJson(res, 200, { service: 'free-model-aggregator', status: 'ok', version: '1.0.0' });
    return;
  }
  if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
    const api = path === '/v1/messages' ? 'anthropic' : 'openai';
    let text: string;
    try {
      text = await readBody(req);
    } catch {
      if (api === 'openai') sendJson(res, 400, openAIError('the request body is too large', 'invalid_request_error', null));
      else sendJson(res, 400, anthropicError('invalid_request_error', 'the request body is too large'));
      return;
    }
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      if (api === 'openai') sendJson(res, 400, openAIError('the request body is not JSON', 'invalid_request_error', null));
      else sendJson(res, 400, anthropicError('invalid_request_error', 'the request body is not JSON'));
      return;
    }
    if (!isRecord(body)) {
      if (api === 'openai') sendJson(res, 400, openAIError('the request body is not one JSON object', 'invalid_request_error', null));
      else sendJson(res, 400, anthropicError('invalid_request_error', 'the request body is not one JSON object'));
      return;
    }
    await handleCompletion(state, api, body, res);
    return;
  }
  sendJson(res, 404, { error: 'not_found', message: `no route ${req.method} ${path}` });
}

/** One registration attempt: POST /register with the catalog. */
async function registerOnce(envUrl: string, port: number, catalog: CatalogDocument): Promise<boolean> {
  try {
    const response = await fetch(`${envUrl.replace(/\/+$/, '')}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) return false;
    const answer: unknown = await response.json();
    return isRecord(answer) && answer.ok === true;
  } catch {
    return false;
  }
}

/** The work poll: done when the env has sent every request. */
async function workIsDone(envUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${envUrl.replace(/\/+$/, '')}/work`, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) return false;
    const work: unknown = await response.json();
    return isRecord(work) && work.done === true;
  } catch {
    return false;
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes('--help') || args.includes('-h')) {
    process.stdout.write(
      'free-model-aggregator: serve the trial env\'s 20 requests with the best free model each.\n' +
        '  node main.ts --list        print the catalog as one JSON document and exit 0\n' +
        '  node main.ts [port]        serve on 0.0.0.0:<port> (an ephemeral port when omitted)\n' +
        'Reads $MAKEFUN_PROMPT_FILE (the brief) and $MAKEFUN_ENV_URL (the env).\n',
    );
    return;
  }
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? '';
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const brief = parseBrief(promptFile);
  if (brief.error !== null) process.stderr.write(`free-model-aggregator: ${brief.error}\n`);
  const catalogs = new Map<string, ProviderCatalog>();
  if (envUrl !== '' && brief.providers.length > 0) {
    const probed = await Promise.all(brief.providers.map((p) => probeProvider(envUrl, p)));
    for (const catalog of probed) {
      if (catalog !== null) catalogs.set(catalog.provider, catalog);
    }
  }
  const catalog = collectCatalog(brief, catalogs);
  if (args.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  const router = new Router(brief, envUrl, catalogs);
  const served: ServedModel[] = freeModelIds(catalog).map((m) => ({ id: m.id, provider: m.provider, display_name: m.id }));
  const state: ServerState = {
    brief,
    envUrl,
    catalog,
    router,
    served,
    nextIndex: 0,
    registered: false,
    exiting: false,
  };
  const portArg = args.find((a) => /^\d+$/.test(a));
  const port = portArg !== undefined ? Number(portArg) : 0;
  const server = createServer((req, res) => {
    handleRequest(state, req, res).catch((err: unknown) => {
      if (!res.headersSent) {
        sendJson(res, 500, { error: 'internal_error', message: err instanceof Error ? err.message : 'unknown error' });
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(port, '0.0.0.0', resolve));
  const address = server.address();
  const actualPort = typeof address === 'object' && address !== null ? address.port : port;
  process.stderr.write(`free-model-aggregator: listening on 0.0.0.0:${actualPort}\n`);
  const shutdown = (): void => {
    if (state.exiting) return;
    state.exiting = true;
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 1000).unref();
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
  if (envUrl === '') {
    // No env to register with or poll: serve until stopped.
    return;
  }
  let lastRegisterAttempt = 0;
  for (;;) {
    if (state.exiting) return;
    const now = Date.now();
    if (!state.registered && now - lastRegisterAttempt >= REGISTER_RETRY_MS) {
      lastRegisterAttempt = now;
      state.registered = await registerOnce(envUrl, actualPort, catalog);
      if (state.registered) process.stderr.write('free-model-aggregator: registered with the env\n');
    }
    if (state.registered && (await workIsDone(envUrl))) {
      process.stderr.write('free-model-aggregator: the work is done, exiting 0\n');
      shutdown();
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, WORK_POLL_MS));
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`free-model-aggregator: fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
