// Upstream wire translation for the free-model aggregator.
//
// The trial's client speaks OpenAI or Anthropic; each provider speaks one of
// openai / anthropic / gemini. The simulated providers echo what they parsed
// (system text, user text, tool names, max_tokens, stop sequences, image hash,
// JSON mode), and the client compares that echo with what it sent, so every
// translation below must carry those fields across wires unchanged:
//
//   openai    system message / user text+image_url / tools / tool_choice /
//             max_tokens / stop / response_format
//   anthropic system field / user text+image / tools / tool_choice /
//             max_tokens / stop_sequences   (no JSON-mode carrier)
//   gemini    systemInstruction / contents parts (text, inlineData, fileData,
//             functionCall, functionResponse) / functionDeclarations /
//             generationConfig (maxOutputTokens, stopSequences,
//             responseMimeType)
//
// A request that needs JSON mode on the OpenAI client wire is therefore never
// routed to an anthropic-wire provider (see main.ts). Tool results are merged
// into the last user turn, never emitted as a trailing turn of their own: the
// providers' parsers take the user text from the last user turn, and a
// trailing tool-only turn would read back as empty text.
//
// Responses are converted the other way, always naming the chosen model:
// usage as the upstream wire reported it, stop reasons mapped across the
// families (stop/end_turn/STOP, length/max_tokens/MAX_TOKENS,
// tool_calls/tool_use, content_filter/refusal/SAFETY). Streaming answers are
// rebuilt from the provider's frames, never re-chunked by byte position:
// the OpenAI stream ends with a usage chunk and [DONE], the Anthropic one
// with message_delta then message_stop.
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, so a replay produces the same bytes.

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every converted OpenAI answer's `created`. */
const FIXED_CREATED = 1767225600;

/** The text of a string, or of a block array's text blocks joined with "\n". */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A JSON value parsed as an object, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringListOf(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.flatMap((v) => (typeof v === 'string' ? [v] : []));
  return [];
}

/** One OpenAI content part as an Anthropic content block, or null. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`. System messages become the system field; tool messages merge their
 * tool_result blocks into the last user message (see the header note).
 */
export function openAIToAnthropicUpstream(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  // The function name of each tool call, by its id, for tool-result turns.
  const callNames = new Map<string, string>();
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'assistant' || !Array.isArray(raw.tool_calls)) continue;
    for (const call of raw.tool_calls) {
      if (isRecord(call) && typeof call.id === 'string' && isRecord(call.function) && typeof call.function.name === 'string') callNames.set(call.id, call.function.name);
    }
  }
  const toolResults: Json[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      toolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role !== 'user') continue;
    const blocks: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const converted = openAIPartToAnthropic(block);
        if (converted !== null) blocks.push(converted);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  // Tool results join the last user message; with no user message they are
  // the only content of a fresh one.
  if (toolResults.length > 0) {
    let target: Json | null = null;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        target = messages[i];
        break;
      }
    }
    if (target === null) {
      target = { role: 'user', content: [] };
      messages.push(target);
    }
    (target.content as Json[]).push(...toolResults);
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringListOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini content part, or null. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** An OpenAI request's tool definitions as Gemini function declarations. */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

function geminiConfigOf(maxTokens: unknown, temperature: unknown, topP: unknown, stops: string[], jsonMode: boolean): Json | null {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return Object.keys(config).length > 0 ? config : null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body (the
 * model travels in the URL, not the body). Tool turns merge their
 * functionResponse parts into the last user content.
 */
export function openAIToGeminiUpstream(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const callNames = new Map<string, string>();
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'assistant' || !Array.isArray(raw.tool_calls)) continue;
    for (const call of raw.tool_calls) {
      if (isRecord(call) && typeof call.id === 'string' && isRecord(call.function) && typeof call.function.name === 'string') callNames.set(call.id, call.function.name);
    }
  }
  const responses: Json[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      responses.push({
        functionResponse: {
          name: callNames.get(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '') ?? '',
          response: { output: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
        },
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const converted = openAIPartToGemini(block);
        if (converted !== null) parts.push(converted);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  if (responses.length > 0) {
    let target: Json | null = null;
    for (let i = contents.length - 1; i >= 0; i--) {
      if (contents[i].role === 'user') {
        target = contents[i];
        break;
      }
    }
    if (target === null) {
      target = { role: 'user', parts: [] };
      contents.push(target);
    }
    (target.parts as Json[]).push(...responses);
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = openAIToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = geminiConfigOf(
    typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens,
    body.temperature,
    body.top_p,
    stringListOf(body.stop),
    format === 'json_object' || format === 'json_schema',
  );
  if (config !== null) out.generationConfig = config;
  return out;
}

/** One Anthropic content block as a Gemini content part, or null. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGeminiUpstream(body: Json): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
              response: { output: toolResultText(block.content) },
            },
          });
          continue;
        }
        const converted = anthropicBlockToGemini(block);
        if (converted !== null) parts.push(converted);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config = geminiConfigOf(body.max_tokens, body.temperature, body.top_p, stringListOf(body.stop_sequences), false);
  if (config !== null) out.generationConfig = config;
  return out;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

// ---------------------------------------------------------------------------
// Response conversion
// ---------------------------------------------------------------------------

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  if (reason === 'TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  if (reason === 'TOOL_CALLS') return 'tool_use';
  return 'end_turn';
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** The text of a Gemini candidate's parts, joined without separator. */
function geminiText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : {};
  if (!Array.isArray(content.parts)) return '';
  return content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** Gemini usageMetadata as OpenAI usage. */
function geminiUsage(ans: Json): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const usage = isRecord(ans.usageMetadata) ? ans.usageMetadata : {};
  const input = numberOrZero(usage.promptTokenCount);
  const output = numberOrZero(usage.candidatesTokenCount);
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer. */
export function openAIJsonFromAnthropic(ans: Json, model: string): Json {
  const content = Array.isArray(ans.content) ? ans.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(ans.usage) ? ans.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: typeof ans.id === 'string' ? ans.id : 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(ans.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions answer. */
export function openAIJsonFromGemini(ans: Json, model: string): Json {
  const candidates = Array.isArray(ans.candidates) ? ans.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const text = geminiText(candidate);
  const usage = geminiUsage(ans);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiStopToOpenAI(candidate.finishReason), logprobs: null }],
    usage,
  };
}

/** A Gemini generateContent answer as an Anthropic Messages answer. */
export function anthropicJsonFromGemini(ans: Json, model: string): Json {
  const candidates = Array.isArray(ans.candidates) ? ans.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const text = geminiText(candidate);
  const usage = geminiUsage(ans);
  return {
    id: `msg_${typeof ans.modelVersion === 'string' && ans.modelVersion !== '' ? ans.modelVersion : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiStopToAnthropic(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

// ---------------------------------------------------------------------------
// SSE handling
// ---------------------------------------------------------------------------

/** One SSE frame: its event name (null when unnamed) and its data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** The SSE frames of a whole event-stream body. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** One server-sent event: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
}

function openAIUsageChunk(model: string, input: number, output: number): string {
  return sseEvent(null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  });
}

/**
 * An Anthropic event stream as OpenAI chunks: role chunk, content and tool
 * deltas, the finish chunk, the usage chunk, then [DONE].
 */
export function openAIChunksFromAnthropicText(text: string, model: string): string {
  const frames = parseSseFrames(text);
  const out: string[] = [];
  let roleSent = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const toolIndexes = new Map<number, number>();
  let nextToolIndex = 0;
  const sendRole = (): void => {
    if (roleSent) return;
    roleSent = true;
    out.push(openAIChunk(model, { role: 'assistant', content: '' }, null));
  };
  for (const frame of frames) {
    if (frame.event === null || frame.event === 'ping') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        sendRole();
        out.push(openAIChunk(model, { content: data.delta.text }, null));
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        const upstreamIndex = typeof data.index === 'number' ? data.index : 0;
        let toolIndex = toolIndexes.get(upstreamIndex);
        if (toolIndex === undefined) {
          toolIndex = nextToolIndex;
          nextToolIndex += 1;
          toolIndexes.set(upstreamIndex, toolIndex);
          sendRole();
          out.push(
            openAIChunk(model, { tool_calls: [{ index: toolIndex, id: '', type: 'function', function: { name: '', arguments: '' } }] }, null),
          );
        }
        out.push(openAIChunk(model, { tool_calls: [{ index: toolIndex, function: { arguments: data.delta.partial_json } }] }, null));
      }
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const upstreamIndex = typeof data.index === 'number' ? data.index : 0;
      if (!toolIndexes.has(upstreamIndex)) {
        const toolIndex = nextToolIndex;
        nextToolIndex += 1;
        toolIndexes.set(upstreamIndex, toolIndex);
        sendRole();
        out.push(
          openAIChunk(model, {
            tool_calls: [
              {
                index: toolIndex,
                id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
                type: 'function',
                function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' },
              },
            ],
          }, null),
        );
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  sendRole();
  out.push(openAIChunk(model, {}, anthropicStopToOpenAI(stopReason ?? 'end_turn')));
  out.push(openAIUsageChunk(model, inputTokens, outputTokens));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/**
 * A Gemini streamGenerateContent stream as OpenAI chunks: role chunk, content
 * deltas, the finish chunk, the usage chunk from the last frame, [DONE].
 */
export function openAIChunksFromGeminiText(text: string, model: string): string {
  const frames = parseSseFrames(text);
  const out: string[] = [];
  let roleSent = false;
  let finishReason: unknown = null;
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const piece = geminiText(candidate);
      if (piece !== '') {
        if (!roleSent) {
          roleSent = true;
          out.push(openAIChunk(model, { role: 'assistant', content: '' }, null));
        }
        out.push(openAIChunk(model, { content: piece }, null));
      }
      if (candidate.finishReason !== undefined) finishReason = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  if (!roleSent) out.push(openAIChunk(model, { role: 'assistant', content: '' }, null));
  out.push(openAIChunk(model, {}, geminiStopToOpenAI(finishReason)));
  out.push(openAIUsageChunk(model, inputTokens, outputTokens));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/**
 * A Gemini streamGenerateContent stream as Anthropic events: message_start
 * (input tokens are unknown until the last frame, so 0; the stream usage
 * check reads the output count from message_delta), text deltas,
 * message_delta with the stop reason and output tokens, message_stop.
 */
export function anthropicEventsFromGeminiText(text: string, model: string): string {
  const frames = parseSseFrames(text);
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
    }),
  );
  out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  let finishReason: unknown = null;
  let outputTokens = 0;
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const piece = geminiText(candidate);
      if (piece !== '') out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
      if (candidate.finishReason !== undefined) finishReason = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
  }
  out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: geminiStopToAnthropic(finishReason), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
  );
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out.join('');
}
