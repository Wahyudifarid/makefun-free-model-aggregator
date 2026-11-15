// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The trial gives this server a brief (provider directory, sealed quality
// table, workload manifest) and an env that serves the providers. The server
// probes every provider's catalog route, builds one catalog, registers its
// port and catalog, then serves the client's 20 requests, one at a time, and
// exits 0 when the env says the workload is done.
//
// ROUTING (quality is 60% of the score, served 15%, failover 10%)
//   The manifest is known up front, so the workload is planned before the
//   first request arrives: each request is assigned the highest-rated free
//   model meeting its requirements, with at most PLAN_CAP planned calls on
//   any one provider, so no provider's hidden free-tier budget is spent on
//   one request family and a failure schedule or outage on one provider can
//   only take a few requests down. Ties break by fewest planned calls, then
//   provider id, then model id: every choice is a function of the brief.
//   A JSON-mode request from the OpenAI client wire is never routed to an
//   anthropic-wire provider: that wire carries no response_format, so the
//   provider would echo json_mode false and the request would fail.
//   On a refusal (402, 429, 5xx, or an unreachable provider) the request is
//   retried on the next candidate, preferring the highest rating among the
//   least-used providers; a provider that has refused twice is treated as
//   spent unless no other candidate remains. Calls are strictly sequential:
//   the client sends one request at a time and this server sends one
//   upstream call at a time, so a provider never sees a concurrent call.
//
// WIRES (10%): every client wire x upstream wire pair is translated in
// lib/translate.ts (and lib/wire.ts for the Anthropic <-> OpenAI pair): the
// system prompt, user text and images, tools, stop sequences, max_tokens
// and JSON mode cross unchanged; usage is reported as the upstream wire
// reported it; stop reasons map across the families. Streams are rebuilt
// from the provider's frames: the OpenAI stream ends with a usage chunk and
// [DONE], the Anthropic one with message_delta then message_stop.
//
// DETERMINISM: no clock, no randomness anywhere in this file or lib/.
// Every decision derives from the brief, the requests and the provider
// answers seen so far, so a replay leaves the same state export.
//
// `node main.ts --list` prints the same catalog JSON the register body
// carries, byte for byte, and exits 0 before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsFromGeminiText,
  anthropicJsonFromGemini,
  anthropicToGeminiUpstream,
  openAIChunksFromAnthropicText,
  openAIChunksFromGeminiText,
  openAIJsonFromAnthropic,
  openAIJsonFromGemini,
  openAIToAnthropicUpstream,
  openAIToGeminiUpstream,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most planned calls on any one provider: spread the workload so one
 *  provider's hidden budget, failure schedule or outage cannot take the
 *  trial down, and leave headroom for failover retries. */
const PLAN_CAP = 3;
/** The most upstream attempts for one client request. */
const MAX_ATTEMPTS = 8;
/** A provider that has refused this many times is treated as spent. */
const SPENT_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** The planned route per manifest request id. */
const planById = new Map<string, Route>();
/** Upstream calls made per provider, failures included: the budget signal. */
const callsUsed = new Map<string, number>();
/** Upstream refusals per provider. */
const failures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  buildPlan();
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can legally serve a request on `clientApi`: it meets
 * the requirements, and its wire can carry what the request needs (an
 * OpenAI JSON-mode request needs a wire with a JSON-mode carrier).
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

/**
 * Candidate order: highest rating first, then fewest calls on that provider,
 * then fewest refusals, then provider id, then model id. Deterministic: no
 * clock, no randomness, only the brief and what the trial has shown so far.
 */
function compareCandidates(a: Candidate, b: Candidate, usage: ReadonlyMap<string, number>, refused: ReadonlyMap<string, number>): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  const ua = usage.get(a.probe.provider.id) ?? 0;
  const ub = usage.get(b.probe.provider.id) ?? 0;
  if (ua !== ub) return ua - ub;
  const fa = refused.get(a.probe.provider.id) ?? 0;
  const fb = refused.get(b.probe.provider.id) ?? 0;
  if (fa !== fb) return fa - fb;
  if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
  return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
}

/** Plan the whole manifest up front: best-rated legal model per request, capped per provider. */
function buildPlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  const planned = new Map<string, number>();
  for (const entry of manifest) {
    const cands = candidatesFor(entry.requires, entry.api).sort((a, b) => compareCandidates(a, b, planned, failures));
    const pick = cands.find((c) => (planned.get(c.probe.provider.id) ?? 0) < PLAN_CAP) ?? cands[0] ?? null;
    if (pick === null) {
      log(`no free model meets ${entry.id}`);
      continue;
    }
    planned.set(pick.probe.provider.id, (planned.get(pick.probe.provider.id) ?? 0) + 1);
    planById.set(entry.id, { probe: pick.probe, model: pick.model });
  }
}

/** The next route to try for a request, excluding providers already tried this request. */
function nextRoute(requires: Requirement, clientApi: 'openai' | 'anthropic', triedProviders: ReadonlySet<string>, triedPairs: ReadonlySet<string>): Route | null {
  const cands = candidatesFor(requires, clientApi);
  if (cands.length === 0) return null;
  const healthy = cands.filter((c) => (failures.get(c.probe.provider.id) ?? 0) < SPENT_FAILURES);
  const pool = healthy.length > 0 ? healthy : cands;
  const untried = pool.filter((c) => !triedProviders.has(c.probe.provider.id));
  const ordered = (untried.length > 0 ? untried : pool).sort((a, b) => compareCandidates(a, b, callsUsed, failures));
  const pick = ordered.find((c) => !triedPairs.has(`${c.probe.provider.id}/${c.model.id}`)) ?? ordered[0] ?? null;
  return pick === null ? null : { probe: pick.probe, model: pick.model };
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
}

/** One upstream provider call in the provider's own wire. Sequential by construction. */
async function callUpstream(route: Route, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    // An Anthropic client body must be rewritten into Chat Completions
    // first: sent as-is, the provider would parse no system prompt, no
    // tools and no stop sequences out of it.
    const out: Json = clientApi === 'openai' ? { ...body, model: route.model.id } : anthropicToOpenAI(body, route.model.id);
    if (stream) {
      const streamOptions = isRecord(body.stream_options) ? body.stream_options : {};
      out.stream_options = { ...streamOptions, include_usage: true };
    }
    upstreamBody = out;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'openai' ? openAIToAnthropicUpstream(body, route.model.id) : { ...body, model: route.model.id };
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    upstreamBody = clientApi === 'openai' ? openAIToGeminiUpstream(body) : anthropicToGeminiUpstream(body);
  }
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const text = await res.text();
    return { ok: res.ok, status: res.status, text };
  } catch (err) {
    log(`upstream ${provider.id} unreachable: ${(err as Error).message}`);
    return { ok: false, status: 0, text: '' };
  }
}

function textToStream(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}

function parseObject(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/**
 * The client's answer from one successful upstream call, converted into the
 * client's wire. Returns false when the upstream answer cannot be converted
 * (the caller then fails over to the next candidate).
 */
async function answerFromUpstream(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  route: Route,
  result: UpstreamResult,
): Promise<boolean> {
  const wire = route.probe.provider.wire;
  const modelId = route.model.id;
  try {
    if (clientApi === 'openai') {
      if (!stream) {
        const answer = parseObject(result.text);
        if (answer === null) return false;
        if (wire === 'openai') sendJson(res, 200, answer);
        else if (wire === 'anthropic') sendJson(res, 200, openAIJsonFromAnthropic(answer, modelId));
        else sendJson(res, 200, openAIJsonFromGemini(answer, modelId));
        return true;
      }
      const sse = wire === 'openai' ? result.text : wire === 'anthropic' ? openAIChunksFromAnthropicText(result.text, modelId) : openAIChunksFromGeminiText(result.text, modelId);
      sendSse(res, sse);
      return true;
    }
    if (!stream) {
      const answer = parseObject(result.text);
      if (answer === null) return false;
      if (wire === 'anthropic') sendJson(res, 200, answer);
      else if (wire === 'openai') sendJson(res, 200, openAIToAnthropic(answer, modelId));
      else sendJson(res, 200, anthropicJsonFromGemini(answer, modelId));
      return true;
    }
    if (wire === 'anthropic') {
      sendSse(res, result.text);
      return true;
    }
    if (wire === 'openai') {
      const events: string[] = [];
      for await (const event of anthropicEventsFromOpenAI(textToStream(result.text), modelId)) events.push(event);
      sendSse(res, events.join(''));
      return true;
    }
    sendSse(res, anthropicEventsFromGeminiText(result.text, modelId));
    return true;
  } catch (err) {
    log(`converting ${wire} answer failed: ${(err as Error).message}`);
    return false;
  }
}

/** Serve one client request: planned route first, then failover over candidates. */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const triedProviders = new Set<string>();
  const triedPairs = new Set<string>();
  let route: Route | null = null;
  if (entry !== undefined) {
    const planned = planById.get(entry.id);
    if (planned !== undefined && (failures.get(planned.probe.provider.id) ?? 0) < SPENT_FAILURES) route = planned;
  }
  let attempts = 0;
  while (attempts < MAX_ATTEMPTS) {
    if (route === null) {
      route = nextRoute(requires, clientApi, triedProviders, triedPairs);
      if (route === null) break;
    }
    attempts += 1;
    const providerId = route.probe.provider.id;
    triedProviders.add(providerId);
    triedPairs.add(`${providerId}/${route.model.id}`);
    const result = await callUpstream(route, clientApi, body, stream);
    callsUsed.set(providerId, (callsUsed.get(providerId) ?? 0) + 1);
    if (result.ok) {
      if (await answerFromUpstream(res, clientApi, stream, route, result)) return;
      log(`upstream ${providerId} answered unusable bytes; failing over`);
    } else {
      log(`upstream ${providerId} refused HTTP ${result.status}; failing over`);
    }
    failures.set(providerId, (failures.get(providerId) ?? 0) + 1);
    route = null;
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
