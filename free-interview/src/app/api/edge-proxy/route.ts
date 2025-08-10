import { NextRequest } from 'next/server';
import { resolveBaseUrl, resolveModel } from '@/lib/llm';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const baseUrl = resolveBaseUrl();
  const model = resolveModel();
  const apiKey = req.headers.get('x-api-key') || process.env.DEEPSEEK_API_KEY || process.env.OPENROUTER_API_KEY;
  if (!apiKey) return new Response(JSON.stringify({ error: 'Missing API key' }), { status: 401 });
  const body = await req.json();
  const upstream = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${apiKey}`,
      'HTTP-Referer': 'https://localhost',
      'X-Title': 'Free Interview',
    },
    body: JSON.stringify({ ...body, model, stream: true }),
  });
  return new Response(upstream.body, {
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    },
  });
}