// Calling providers and choosing them. The router works from the brief's
// quality table and from the providers' own catalogs (probed once at start):
// for each request it picks the best free model that meets the request's
// requirements, on the wire the request's JSON mode allows, and on a 402/429
// it moves on to the next candidate. Calls to one provider are sequential —
// the providers charge a 429 for a concurrent call, so they never overlap.
//
// Determinism: candidates are ordered by rating and then by the brief's
// provider and model order; a retry never revisits a failed (provider,
// model) within one request; a provider is only set aside after three
// refusals in a row, which prove its budget spent or an outage begun.
// Nothing here reads the clock or random state.

import {
  anthropicToGemini,
  anthropicToOpenAI,
  geminiJsonToAnthropic,
  geminiJsonToOpenAI,
  openAIJsonToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  anthropicJsonToOpenAI,
  parseSseFrames,
  geminiSseToAnthropic,
  geminiSseToOpenAI,
  openAISseToAnthropic,
  anthropicSseToOpenAI,
} from './wire.ts';
import {
  ANTHROPIC_VERSION,
  authHeaders,
  joinBase,
  keyFor,
  type ClientRequest,
  type ProviderCatalog,
  type QualityModel,
  type Provider,
  type Brief,
  type Json,
  isRecord,
} from './brief.ts';

/** One candidate model for a request, in preference order. */
export interface Candidate {
  modelId: string;
  providerId: string;
  rating: number;
  model: QualityModel;
  wire: 'openai' | 'anthropic' | 'gemini';
  auth: { scheme: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none'; header: string; key: string };
  /** The provider's base path (`/p/<id>`), joined to the env URL when dialling. */
  base: string;
}

/** What a provider call returns: the status, a parsed JSON body when the
 * answer was JSON, the raw text when it was an SSE stream, and whether a
 * retry on another provider is in order. */
export interface ProviderResult {
  status: number;
  body: Json | null;
  text: string;
  stream: boolean;
  retriable: boolean;
}

/** POST one JSON body to a URL with a per-call timeout (10 s). `stream` says
 * whether the answer comes back as server-sent events: it is the client's
 * stream flag, not a field of the translated body — the Gemini wire carries
 * streaming in the URL, not the body, so reading it off the body would
 * mistake a Gemini stream for JSON. */
export async function postJson(
  url: string,
  headers: Record<string, string>,
  body: Json,
  stream: boolean,
  timeoutMs = 10_000,
): Promise<ProviderResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    if (response.status !== 200) {
      return {
        status: response.status,
        body: null,
        text,
        stream: false,
        retriable: response.status === 402 || response.status === 429 || response.status >= 500,
      };
    }
    if (stream) {
      return { status: 200, body: null, text, stream: true, retriable: false };
    }
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed)) throw new Error('provider answer was not a JSON object');
    return { status: 200, body: parsed, text, stream: false, retriable: false };
  } catch {
    return { status: 0, body: null, text: '', stream: false, retriable: true };
  } finally {
    clearTimeout(timer);
  }
}

/** The client's body translated for one candidate's wire. */
function upstreamBodyFor(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Json {
  if (candidate.wire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model: candidate.modelId };
    return openAIToAnthropic(body, candidate.modelId);
  }
  if (candidate.wire === 'gemini') {
    if (clientApi === 'anthropic') return anthropicToGemini(body);
    return openAIToGemini(body);
  }
  if (clientApi === 'anthropic') return anthropicToOpenAI(body, candidate.modelId);
  const out: Json = { ...body, model: candidate.modelId };
  // A streamed OpenAI answer must carry a usage chunk, and the provider only
  // sends one when the request asks for it.
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** The provider's URL for one candidate: the env URL plus its base path, plus the upstream wire's path. */
function upstreamUrlFor(envUrl: string, candidate: Candidate, stream: boolean): string {
  const base = joinBase(envUrl, candidate.base);
  if (candidate.wire === 'gemini') {
    return `${base}/v1beta/models/${candidate.modelId}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  }
  if (candidate.wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1/chat/completions`;
}

/** The headers one candidate's calls carry: its key, plus anthropic-version on the Anthropic wire. */
function upstreamHeadersFor(candidate: Candidate): Record<string, string> {
  const headers = authHeaders(candidate.auth.scheme, candidate.auth.key);
  if (candidate.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  return headers;
}

/** The upstream request for one candidate: URL, headers, translated body. */
export function upstreamRequestFor(
  envUrl: string,
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): { url: string; headers: Record<string, string>; body: Json } {
  return {
    url: upstreamUrlFor(envUrl, candidate, body.stream === true),
    headers: upstreamHeadersFor(candidate),
    body: upstreamBodyFor(candidate, clientApi, body),
  };
}

/** The client request translated back from an upstream JSON answer. */
export function translateJsonAnswer(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  answer: Json,
): Json {
  if (candidate.wire === 'openai') {
    return clientApi === 'openai' ? answer : openAIJsonToAnthropic(answer, candidate.modelId);
  }
  if (candidate.wire === 'anthropic') {
    return clientApi === 'anthropic' ? answer : anthropicJsonToOpenAI(answer, candidate.modelId);
  }
  return clientApi === 'openai'
    ? geminiJsonToOpenAI(answer, candidate.modelId)
    : geminiJsonToAnthropic(answer, candidate.modelId);
}

/** The client request translated back from an upstream SSE stream. */
export function translateSseAnswer(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  text: string,
  usage: { input: number; output: number },
): string {
  const frames = parseSseFrames(text);
  if (candidate.wire === 'openai') {
    return clientApi === 'openai' ? text : openAISseToAnthropic(frames, candidate.modelId, usage);
  }
  if (candidate.wire === 'anthropic') {
    return clientApi === 'anthropic' ? text : anthropicSseToOpenAI(frames, candidate.modelId, usage);
  }
  return clientApi === 'openai'
    ? geminiSseToOpenAI(frames, candidate.modelId, usage)
    : geminiSseToAnthropic(frames, candidate.modelId, usage);
}

/** Whether a model meets a request's requirements. */
export function meetsRequirements(model: QualityModel, requirements: Json): boolean {
  const minContext = typeof requirements.min_context === 'number' ? requirements.min_context : null;
  if (minContext !== null && (model.context === null || model.context < minContext)) return false;
  if (requirements.tools === true && !model.tools) return false;
  if (requirements.vision === true && !model.vision) return false;
  if (requirements.json === true && !model.json) return false;
  return true;
}

/** The router: the catalogs probed at start, the brief's quality table, and
 * the per-provider failure state. A 402/429 is not, on its own, the end of a
 * provider: its failure schedule is one or two single call numbers, so the
 * next call may pass — only three refusals in a row prove the budget is
 * spent or an outage has begun (a provider has at most two scheduled
 * failures), and only then is it set aside for the rest of the trial. */
export class Router {
  private readonly brief: Brief;
  private readonly envUrl: string;
  private readonly providers: Map<string, { wire: 'openai' | 'anthropic' | 'gemini'; auth: Provider['auth']; key: string; base: string }> = new Map();
  /** Refusals in a row per provider; a success resets it. */
  private readonly consecutiveFailures: Map<string, number> = new Map();
  /** Providers proven spent or out: three refusals in a row. */
  private readonly dead: Set<string> = new Set();

  constructor(brief: Brief, envUrl: string, catalogs: Map<string, ProviderCatalog>) {
    this.brief = brief;
    this.envUrl = envUrl;
    for (const provider of brief.providers) {
      const catalog = catalogs.get(provider.id);
      if (catalog === undefined) continue;
      this.providers.set(provider.id, {
        wire: provider.wire,
        auth: provider.auth,
        key: keyFor(provider),
        base: provider.base,
      });
    }
  }

  /** Whether a provider is out of calls for this trial (proven spent or out). */
  isExhausted(providerId: string): boolean {
    return this.dead.has(providerId);
  }

  /** The candidates for a request, best first: free models that meet its
   * requirements, on the wire its JSON mode allows, on providers that have
   * not been proven spent yet. */
  candidatesFor(request: ClientRequest, jsonMode: boolean): Candidate[] {
    const out: Candidate[] = [];
    for (const provider of this.brief.providers) {
      if (this.dead.has(provider.id)) continue;
      const router = this.providers.get(provider.id);
      if (router === undefined) continue;
      const models = this.brief.qualityTable[provider.id] ?? [];
      for (const model of models) {
        if (!model.free) continue;
        if (!meetsRequirements(model, request.requirements)) continue;
        if (jsonMode && router.wire === 'anthropic') continue;
        out.push({
          modelId: model.id,
          providerId: provider.id,
          rating: model.rating,
          model,
          wire: router.wire,
          auth: { scheme: router.auth.scheme, header: router.auth.header, key: router.key },
          base: router.base,
        });
      }
    }
    const providerOrder = new Map(this.brief.providers.map((p, i) => [p.id, i]));
    const modelOrder = new Map<string, number>();
    for (const provider of this.brief.providers) {
      (this.brief.qualityTable[provider.id] ?? []).forEach((m, i) => modelOrder.set(`${provider.id}/${m.id}`, i));
    }
    out.sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      const aProvider = providerOrder.get(a.providerId) ?? 0;
      const bProvider = providerOrder.get(b.providerId) ?? 0;
      if (aProvider !== bProvider) return aProvider - bProvider;
      return (modelOrder.get(`${a.providerId}/${a.modelId}`) ?? 0) - (modelOrder.get(`${b.providerId}/${b.modelId}`) ?? 0);
    });
    return out;
  }

  /**
   * One provider call: the translated request, the provider's answer, and
   * the status. A 402/429 counts as one refusal in a row for that provider
   * (three in a row prove it spent or out); a 200 resets the count. A 5xx
   * or network error is retriable on the next candidate and counts for
   * neither; any other non-200 is not retriable.
   *
   * A refusal is retried once on the same model before giving up on it: a
   * provider's failure schedule is one or two single call numbers, so the
   * next call — a new call number — usually passes, and the retry keeps the
   * request on the best model instead of falling back. A refusal that is a
   * spent budget or an outage fails the retry too, and the caller moves on.
   */
  async call(
    candidate: Candidate,
    clientApi: 'openai' | 'anthropic',
    body: Json,
  ): Promise<{ body: Json | null; text: string; stream: boolean; retriable: boolean; status: number }> {
    for (let attempt = 0; attempt < 2; attempt++) {
      const request = upstreamRequestFor(this.envUrl, candidate, clientApi, body);
      const result = await postJson(request.url, request.headers, request.body, body.stream === true);
      if (result.status === 402 || result.status === 429) {
        const count = (this.consecutiveFailures.get(candidate.providerId) ?? 0) + 1;
        this.consecutiveFailures.set(candidate.providerId, count);
        if (count >= 3) this.dead.add(candidate.providerId);
      } else if (result.status === 200) {
        this.consecutiveFailures.set(candidate.providerId, 0);
      }
      const out = {
        body: result.body,
        text: result.text,
        stream: result.stream,
        retriable: result.retriable,
        status: result.status,
      };
      if (result.status === 200) return out;
      if (result.status !== 402 && result.status !== 429) return out;
      if (attempt === 1) return out;
      // One refusal on this model: the next call number may pass. Retry it once.
    }
    return { body: null, text: '', stream: false, retriable: true, status: 0 };
  }
}
