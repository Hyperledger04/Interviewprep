import { NextRequest } from 'next/server';
import { llmChat } from '@/lib/llm';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const { messages } = await req.json();
  if (!Array.isArray(messages)) {
    return new Response(JSON.stringify({ error: 'messages array required' }), { status: 400 });
  }
  const res = await llmChat(messages, { stream: true }, req);
  return new Response(res.body, {
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    },
  });
}