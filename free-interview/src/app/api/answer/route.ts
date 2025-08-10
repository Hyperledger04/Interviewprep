import { NextRequest } from 'next/server';
import { llmChat } from '@/lib/llm';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const { question, cvText, jdText } = await req.json();
  if (!question) {
    return new Response(JSON.stringify({ error: 'question required' }), { status: 400 });
  }

  const system = {
    role: 'system' as const,
    content: 'You are a helpful assistant. Provide a concise, structured, high-quality answer grounded in the CV and JD context. Use bullet points and crisp sentences.',
  };
  const user = {
    role: 'user' as const,
    content: `Question: ${question}\n\nCV Context (optional):\n${cvText || ''}\n\nJD Context (optional):\n${jdText || ''}`,
  };

  const res = await llmChat([system, user], { stream: true }, req);
  return new Response(res.body, {
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    },
  });
}