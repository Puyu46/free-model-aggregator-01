// The client wires and the three upstream wires, and the translations between
// them. Every function here is pure: the same body and the same provider
// answer give the same bytes on a replay. No clock, no randomness; the two
// model listings use fixed timestamps for the same reason.
//
// What the translations must preserve (the client compares the provider's
// echo with what it sent): the system text, the last user message's text, the
// tool names in order, max_tokens, the stop sequences, the image bytes, and
// JSON mode. A field rewritten or dropped loses the request.
//
// The providers always answer with plain text and a family-0 stop reason
// ("stop" / "end_turn" / "STOP"); they never emit tool calls or truncate on
// max_tokens. The mappings below still handle those shapes, for completeness.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
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

/** The provider's usage for one answer, from the client's own request and the
 * answer text — the same formula the providers use, so the numbers the client
 * is handed match the numbers the provider recorded. */
export function usageOf(
  clientBody: Json,
  clientApi: 'openai' | 'anthropic',
  answerText: string,
): { input: number; output: number } {
  let systemText = '';
  let userText = '';
  const tools: string[] = [];
  if (clientApi === 'openai') {
    const messages = Array.isArray(clientBody.messages) ? clientBody.messages : [];
    const systemParts: string[] = [];
    for (const raw of messages) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemParts.push(text);
      } else if (raw.role === 'user') {
        userText = textOf(raw.content);
      }
    }
    systemText = systemParts.join('\n');
    if (Array.isArray(clientBody.tools)) {
      for (const tool of clientBody.tools) {
        if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') {
          tools.push(tool.function.name);
        }
      }
    }
  } else {
    systemText = textOf(clientBody.system);
    const messages = Array.isArray(clientBody.messages) ? clientBody.messages : [];
    for (const raw of messages) {
      if (!isRecord(raw) || raw.role !== 'user') continue;
      userText = textOf(raw.content);
    }
    if (Array.isArray(clientBody.tools)) {
      for (const tool of clientBody.tools) {
        if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
      }
    }
  }
  const prompt = systemText.length + userText.length + tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Request translations: the client's body as an upstream body that names `model`.
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
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

/** One OpenAI content part as an Anthropic content block; null for the rest. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return {
        type: 'image',
        source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] },
      };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** A stop list as the env reads it: a string counts as one stop, an array as its string members. */
function stopList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** The system text of an OpenAI body: every system message, joined with "\n". */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** OpenAI tools as Anthropic tools. */
function openAIToolsToAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters)
          ? raw.function.parameters
          : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as OpenAI tools. */
function anthropicToolsToOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

/** OpenAI tools as Gemini function declarations. */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters)
          ? raw.function.parameters
          : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini function declarations. */
function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`. The response_format is dropped: the Anthropic wire cannot
 * carry JSON mode, so a JSON-mode request is never routed to this wire.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToBlock(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  const system = openAISystemText(body);
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsToAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  if (body.tool_choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (body.tool_choice === 'required') out.tool_choice = { type: 'any' };
  else if (body.tool_choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(body.tool_choice) && body.tool_choice.type === 'function' && isRecord(body.tool_choice.function)) {
    out.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  const tools = anthropicToolsToOpenAI(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI message as Gemini contents; tool messages are skipped (the
 * provider's echo never reads them: only the last user content's text). */
function openAIMessageToGeminiParts(raw: Json): { role: string; parts: Json[] } | null {
  if (raw.role === 'tool') return null;
  const role = raw.role === 'assistant' ? 'model' : 'user';
  const parts: Json[] = [];
  if (typeof raw.content === 'string') {
    if (raw.content !== '') parts.push({ text: raw.content });
  } else if (Array.isArray(raw.content)) {
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) {
          parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        } else {
          parts.push({ fileData: { fileUri: url } });
        }
      }
    }
  }
  if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
    for (const call of raw.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      parts.push({
        functionCall: {
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: argumentsOf(call.function.arguments),
        },
      });
    }
  }
  if (parts.length === 0) return null;
  return { role, parts };
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    const content = openAIMessageToGeminiParts(raw);
    if (content !== null) contents.push(content);
  }
  const out: Json = { contents };
  const system = openAISystemText(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = openAIToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const openAIStops = stopList(body.stop);
  if (openAIStops.length > 0) config.stopSequences = openAIStops;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({
              inlineData: {
                mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
                data: block.source.data,
              },
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: { name: '', response: { content: toolResultText(block.content) } },
          });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = anthropicToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const anthropicStops = stopList(body.stop_sequences);
  if (anthropicStops.length > 0) {
    config.stopSequences = anthropicStops;
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response translations: an upstream JSON answer in the client's wire.
// ---------------------------------------------------------------------------

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIJsonToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicJsonToOpenAI(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = textOf(content);
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : '',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text !== '' ? text : null };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** The text of a Gemini answer: every candidate part's text, joined. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const parts: string[] = [];
  for (const candidate of candidates) {
    if (!isRecord(candidate) || !isRecord(candidate.content)) continue;
    const list = Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
    for (const part of list) {
      if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
    }
  }
  return parts.join('');
}

/** A Gemini answer's finishReason, or null. */
function geminiFinish(body: Json): unknown {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return first.finishReason ?? null;
}

/** A Gemini answer's usage, as input/output token counts. */
function geminiUsage(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiJsonToOpenAI(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: geminiText(body) },
        finish_reason: geminiStopToOpenAI(geminiFinish(body)),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiJsonToAnthropic(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: geminiStopToAnthropic(geminiFinish(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

// ---------------------------------------------------------------------------
// SSE: parse frames, then rebuild the stream in the client's wire.
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

/** One SSE text as its frames: event names and data lines, in order. */
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

/** The text an OpenAI SSE stream carried: every content delta, in order. */
export function openAIStreamText(text: string): string {
  return collectOpenAIStream(parseSseFrames(text)).text;
}

/** The text an Anthropic SSE stream carried: every text delta, in order. */
export function anthropicStreamText(text: string): string {
  return collectAnthropicStream(parseSseFrames(text)).text;
}

/** The text a Gemini SSE stream carried: every text part, in order. */
export function geminiStreamText(text: string): string {
  return collectGeminiStream(parseSseFrames(text)).text;
}

/** The text an upstream SSE stream carried, in its own wire. */
export function upstreamStreamText(wire: 'openai' | 'anthropic' | 'gemini', text: string): string {
  if (wire === 'openai') return openAIStreamText(text);
  if (wire === 'anthropic') return anthropicStreamText(text);
  return geminiStreamText(text);
}

/** One server-sent event: `event: <name>` when named, then the data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One OpenAI chunk: a delta, a finish_reason, or (choices empty) a usage chunk. */
function openAIChunk(
  model: string,
  delta: Json | null,
  finish: string | null,
  usage: { input: number; output: number } | null,
): string {
  if (usage !== null) {
    return sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        total_tokens: usage.input + usage.output,
      },
    });
  }
  return sseEvent(null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta: delta ?? {}, finish_reason: finish }],
  });
}

/**
 * An OpenAI SSE stream with a usage chunk in it: a stream that already
 * carries one is returned untouched; one that does not gets a usage chunk
 * inserted before the [DONE] line (or appended when there is none). The
 * client requires the final usage in a streamed answer, and a provider only
 * sends it when the request asked for it — this is the backstop for a
 * same-wire stream that went through untouched.
 */
export function ensureOpenAIUsageChunk(
  text: string,
  model: string,
  usage: { input: number; output: number },
): string {
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    try {
      const chunk: unknown = JSON.parse(frame.data);
      if (isRecord(chunk) && isRecord(chunk.usage)) return text;
    } catch {
      // Not JSON: look on.
    }
  }
  const usageChunk = openAIChunk(model, null, null, usage);
  const doneAt = text.indexOf('data: [DONE]');
  if (doneAt === -1) return `${text}${usageChunk}`;
  return `${text.slice(0, doneAt)}${usageChunk}${text.slice(doneAt)}`;
}

interface CollectedText {
  text: string;
  /** Tool calls in order: id, name, and the arguments text accumulated so far. */
  tools: { id: string; name: string; args: string }[];
  stop: string | null;
}

/** The text, tool calls and stop reason an OpenAI SSE stream carried. */
function collectOpenAIStream(frames: SseFrame[]): CollectedText {
  const out: CollectedText = { text: '', tools: [], stop: null };
  for (const frame of frames) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out.text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : out.tools.length;
      let tool = out.tools[index];
      if (tool === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        tool = {
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${index}`,
          name: typeof fn.name === 'string' ? fn.name : '',
          args: '',
        };
        out.tools[index] = tool;
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string') tool.args += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') out.stop = choice.finish_reason;
  }
  return out;
}

/** The text, tool calls and stop reason an Anthropic SSE stream carried. */
function collectAnthropicStream(frames: SseFrame[]): CollectedText {
  const out: CollectedText = { text: '', tools: [], stop: null };
  const byIndex = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of frames) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'content_block_start' && isRecord(data.content_block)) {
      const index = typeof data.index === 'number' ? data.index : -1;
      const block = data.content_block;
      if (block.type === 'tool_use') {
        byIndex.set(index, {
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          args: '',
        });
      }
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : -1;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        out.text += data.delta.text;
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const tool = byIndex.get(index);
        if (tool !== undefined) tool.args += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta' && isRecord(data.delta)) {
      if (typeof data.delta.stop_reason === 'string') out.stop = data.delta.stop_reason;
    }
  }
  out.tools = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, tool]) => tool);
  return out;
}

/** The text, finish reason and usage a Gemini SSE stream carried. */
function collectGeminiStream(frames: SseFrame[]): {
  text: string;
  stop: unknown;
  usage: { input: number; output: number } | null;
} {
  let text = '';
  let stop: unknown = null;
  let usage: { input: number; output: number } | null = null;
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    for (const candidate of candidates) {
      if (!isRecord(candidate)) continue;
      if (isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
        for (const part of candidate.content.parts) {
          if (isRecord(part) && typeof part.text === 'string') text += part.text;
        }
      }
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      usage = {
        input: numberOrZero(data.usageMetadata.promptTokenCount),
        output: numberOrZero(data.usageMetadata.candidatesTokenCount),
      };
    }
  }
  return { text, stop, usage };
}

/** Split text into the provider's 13-character stream pieces. */
function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/**
 * An OpenAI SSE stream as the Anthropic event sequence: message_start, one
 * text block (then one tool_use block per call), message_delta with the stop
 * reason and the output tokens, message_stop. `usage` is the provider's
 * usage for the answer, from the client's request and the answer text.
 */
export function openAISseToAnthropic(
  frames: SseFrame[],
  model: string,
  usage: { input: number; output: number },
): string {
  const got = collectOpenAIStream(frames);
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  );
  let nextIndex = 0;
  if (got.text !== '') {
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: nextIndex,
        content_block: { type: 'text', text: '' },
      }),
    );
    for (const piece of splitPieces(got.text)) {
      out.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index: nextIndex,
          delta: { type: 'text_delta', text: piece },
        }),
      );
    }
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: nextIndex }));
    nextIndex += 1;
  }
  for (const tool of got.tools) {
    const index = nextIndex;
    nextIndex += 1;
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: tool.id, name: tool.name, input: {} },
      }),
    );
    if (tool.args !== '') {
      out.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: tool.args },
        }),
      );
    }
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: openAIStopToAnthropic(got.stop), stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}

/**
 * An Anthropic SSE stream as OpenAI chunks: a role chunk, content deltas,
 * tool call deltas, a finish chunk, a usage chunk, then [DONE].
 */
export function anthropicSseToOpenAI(
  frames: SseFrame[],
  model: string,
  usage: { input: number; output: number },
): string {
  const got = collectAnthropicStream(frames);
  const out: string[] = [];
  out.push(openAIChunk(model, { role: 'assistant', content: '' }, null, null));
  for (const piece of splitPieces(got.text)) {
    out.push(openAIChunk(model, { content: piece }, null, null));
  }
  got.tools.forEach((tool, index) => {
    out.push(
      openAIChunk(
        model,
        {
          tool_calls: [
            {
              index,
              id: tool.id,
              type: 'function',
              function: { name: tool.name, arguments: '' },
            },
          ],
        },
        null,
        null,
      ),
    );
    if (tool.args !== '') {
      out.push(
        openAIChunk(
          model,
          { tool_calls: [{ index, function: { arguments: tool.args } }] },
          null,
          null,
        ),
      );
    }
  });
  out.push(openAIChunk(model, {}, anthropicStopToOpenAI(got.stop), null));
  out.push(openAIChunk(model, null, null, usage));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini SSE stream as OpenAI chunks. */
export function geminiSseToOpenAI(
  frames: SseFrame[],
  model: string,
  usage: { input: number; output: number },
): string {
  const got = collectGeminiStream(frames);
  const out: string[] = [];
  out.push(openAIChunk(model, { role: 'assistant', content: '' }, null, null));
  for (const piece of splitPieces(got.text)) {
    out.push(openAIChunk(model, { content: piece }, null, null));
  }
  out.push(openAIChunk(model, {}, geminiStopToOpenAI(got.stop), null));
  out.push(openAIChunk(model, null, null, usage));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini SSE stream as the Anthropic event sequence. */
export function geminiSseToAnthropic(
  frames: SseFrame[],
  model: string,
  usage: { input: number; output: number },
): string {
  const got = collectGeminiStream(frames);
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  );
  if (got.text !== '') {
    out.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      }),
    );
    for (const piece of splitPieces(got.text)) {
      out.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'text_delta', text: piece },
        }),
      );
    }
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  }
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: geminiStopToAnthropic(got.stop), stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}
