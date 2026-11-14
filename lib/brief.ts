// The brief: the three JSON documents the prompt file carries, and the
// catalog probe. The prompt file is prose with three JSON documents inside
// it, found by their `schema` member: the provider directory
// (`free-model-aggregator-01/providers/v1`), the sealed quality table
// (`free-model-aggregator-01/ratings/v1`) and the workload manifest
// (`free-model-aggregator-01/workload/v1`). The prose does not matter.
//
// The catalog this builds is the document `--list` prints and `/register`
// posts, byte for byte: one entry a provider — id, base_url (the directory's
// own path, `/p/<id>`, never joined to $MAKEFUN_ENV_URL), key_env, wire,
// and models, each with id, free and context_length. The models come from
// the provider's own catalog route (the truth the catalog is scored
// against); the quality table only decides the free flag where it has a row.
// A probe that fails leaves its provider out of the catalog and it gets no
// traffic; the trial goes on.
//
// Everything here is deterministic: no clock, no randomness.

import { readFileSync } from 'node:fs';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

/** The header that carries a provider's key, or '' when it sends none. */
export function authHeaderName(scheme: AuthScheme): string {
  if (scheme === 'bearer') return 'Authorization';
  if (scheme === 'x-api-key') return 'x-api-key';
  if (scheme === 'x-goog-api-key') return 'x-goog-api-key';
  return '';
}

/** The headers a provider wants its key in; none when the scheme is 'none' or the key is empty. */
export function authHeaders(scheme: AuthScheme, key: string): Record<string, string> {
  if (key === '' || scheme === 'none') return {};
  if (scheme === 'bearer') return { Authorization: `Bearer ${key}` };
  return { [authHeaderName(scheme)]: key };
}

export const ANTHROPIC_VERSION = '2023-06-01';

/** One provider as the brief lists it. `base` is the base path (`/p/<id>`) its routes hang off. */
export interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: { scheme: AuthScheme; header: string; envVar: string };
  /** The key the simulated service takes, as the brief lists it. */
  key: string;
  base: string;
}

/** One row of the sealed quality table. */
export interface QualityModel {
  id: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** One request the client will send, in manifest order. */
export interface ClientRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requirements: Json;
}

/** The brief as read: the directory, the quality table, the manifest. */
export interface Brief {
  providers: Provider[];
  /** The quality table by provider id, in the brief's order. */
  qualityTable: Record<string, QualityModel[]>;
  /** The best free rating per request id, from the table's `best` list. */
  best: Record<string, number>;
  requests: ClientRequest[];
  error: string | null;
}

/** One model as a provider's catalog route listed it. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider's catalog as probed: its directory row and the models its route listed. */
export interface ProviderCatalog {
  provider: string;
  wire: Wire;
  baseUrl: string;
  keyEnv: string;
  models: CatalogModel[];
}

/** The catalog document `--list` prints and `/register` posts. */
export interface CatalogDocument {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: Wire;
    models: CatalogModel[];
  }[];
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The key to dial a provider with: the environment variable when it is set, else the brief's key. */
export function keyFor(provider: Provider): string {
  const envVar = provider.auth.envVar;
  if (envVar !== '') {
    const fromEnv = process.env[envVar];
    if (fromEnv !== undefined && fromEnv !== '') return fromEnv;
  }
  return provider.key;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function providerOf(v: unknown): Provider | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const scheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return {
    id,
    name: stringOf(v.name) !== '' ? stringOf(v.name) : id,
    wire,
    auth: { scheme, header: authHeaderName(scheme), envVar: stringOf(v.key_env) },
    key: stringOf(v.key),
    base,
  };
}

function qualityModelOf(v: unknown): QualityModel | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (id === '' || provider === '') return null;
  return {
    id,
    provider,
    rating: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function requestOf(v: unknown): ClientRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requirements: isRecord(v.requires) ? v.requires : {},
  };
}

/** The brief at `path`, its three documents found by schema. */
export function parseBrief(path: string): Brief {
  const empty: Brief = { providers: [], qualityTable: {}, best: {}, requests: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const qualityTable: Record<string, QualityModel[]> = {};
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)) {
    for (const raw of ratingsDoc.ratings) {
      const row = qualityModelOf(raw);
      if (row === null) continue;
      (qualityTable[row.provider] ??= []).push(row);
    }
  }
  const best: Record<string, number> = {};
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const raw of ratingsDoc.best) {
      if (!isRecord(raw)) continue;
      const id = stringOf(raw.id);
      if (id === '') continue;
      best[id] = typeof raw.rating_milli === 'number' ? raw.rating_milli : 0;
    }
  }
  const requests: ClientRequest[] = [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  if (workloadDoc !== null && Array.isArray(workloadDoc.manifest)) {
    for (const raw of workloadDoc.manifest) {
      const request = requestOf(raw);
      if (request !== null) requests.push(request);
    }
  }
  return { providers, qualityTable, best, requests, error: null };
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): CatalogModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return { id, free: boolOrNull(v.free) ?? false, context_length: context };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): CatalogModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/**
 * Probe one provider's catalog route, in its own wire, tolerating every
 * failure: a failed probe leaves the provider out of the catalog.
 */
export async function probeProvider(envUrl: string, provider: Provider, timeoutMs = 5000): Promise<ProviderCatalog | null> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth.scheme, keyFor(provider)) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const response = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!response.ok) return null;
    const body: unknown = await response.json();
    return {
      provider: provider.id,
      wire: provider.wire,
      baseUrl: provider.base,
      keyEnv: provider.auth.envVar,
      models: modelsIn(provider.wire, body),
    };
  } catch {
    return null;
  }
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, models sorted by id. The free flag comes
 * from the quality table's row where it has one (the sealed truth), else the
 * listing's own word; the context length the same way. Its base_url is the
 * directory's own path, never the path joined to envUrl.
 */
export function collectCatalog(brief: Brief, catalogs: Map<string, ProviderCatalog>): CatalogDocument {
  const providers: CatalogDocument['providers'] = [];
  for (const provider of brief.providers) {
    const catalog = catalogs.get(provider.id);
    if (catalog === undefined) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...catalog.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = (brief.qualityTable[provider.id] ?? []).find((m) => m.id === model.id) ?? null;
      models.push({
        id: model.id,
        free: row !== null ? row.free : model.free,
        context_length: row?.context ?? model.context_length ?? null,
      });
    }
    providers.push({
      id: provider.id,
      base_url: provider.base,
      key_env: provider.auth.envVar,
      wire: provider.wire,
      models,
    });
  }
  return { providers };
}

/** The free model ids the catalog lists, in catalog order — what GET /v1/models answers. */
export function freeModelIds(catalog: CatalogDocument): { id: string; provider: string }[] {
  const out: { id: string; provider: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free) out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}
