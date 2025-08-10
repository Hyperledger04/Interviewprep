import { NextRequest } from 'next/server';

export type ChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'developer' | 'tool';
  content: string;
};

export type LlmOptions = {
  temperature?: number;
  response_format?: 'json_object' | 'text';
  reasoning?: boolean;
  max_output_tokens?: number;
  stream?: boolean;
};

type ChatCompletionsResponse = {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
};

export function resolveApiKey(req?: NextRequest): string | undefined {
  const headerKey = req?.headers.get('x-api-key') || req?.headers.get('authorization')?.replace(/^Bearer\s+/i, '');
  const envKey = process.env.DEEPSEEK_API_KEY || process.env.OPENROUTER_API_KEY;
  return headerKey || envKey;
}

export function resolveBaseUrl(): string {
  return process.env.LLM_BASE_URL || 'https://openrouter.ai/api/v1';
}

export function resolveModel(): string {
  return process.env.LLM_MODEL || 'deepseek/deepseek-r1:free';
}

export async function llmChat(
  messages: ChatMessage[],
  options: LlmOptions = {},
  req?: NextRequest
): Promise<Response> {
  const apiKey = resolveApiKey(req);
  if (!apiKey) {
    return new Response(JSON.stringify({ error: 'Missing API key. Provide via .env or header x-api-key' }), {
      status: 401,
      headers: { 'content-type': 'application/json' },
    });
  }

  const baseUrl = resolveBaseUrl();
  const model = resolveModel();

  const body: Record<string, unknown> = {
    model,
    messages,
    temperature: options.temperature ?? 0.7,
    stream: options.stream ?? false,
    max_tokens: options.max_output_tokens ?? undefined,
  };

  if (options.response_format === 'json_object') {
    body.response_format = { type: 'json_object' };
  }

  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${apiKey}`,
      'HTTP-Referer': 'https://localhost',
      'X-Title': 'Free Interview',
    },
    body: JSON.stringify(body),
  });

  return res;
}

export async function llmJson<T = unknown>(
  messages: ChatMessage[],
  req?: NextRequest,
  temperature = 0.3
): Promise<T> {
  const res = await llmChat(messages, { response_format: 'json_object', stream: false, temperature }, req);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`LLM error: ${res.status} ${text}`);
  }
  const data = (await res.json()) as ChatCompletionsResponse;
  const content = data?.choices?.[0]?.message?.content;
  if (!content) throw new Error('Empty LLM response');
  try {
    return JSON.parse(content) as T;
  } catch {
    const match = String(content).match(/\{[\s\S]*\}/);
    if (match) {
      return JSON.parse(match[0]) as T;
    }
    throw new Error('Failed to parse JSON from LLM response');
  }
}