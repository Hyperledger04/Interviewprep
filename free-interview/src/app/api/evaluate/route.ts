import { NextRequest } from 'next/server';
import { llmJson } from '@/lib/llm';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export type Evaluation = {
  score: number; // 0-10
  strengths: string[];
  improvements: string[];
  summary: string;
  suggested_answer: string;
};

export async function POST(req: NextRequest) {
  const { question, candidateAnswer, cvText, jdText } = await req.json();
  if (!question || !candidateAnswer) {
    return new Response(JSON.stringify({ error: 'question and candidateAnswer required' }), { status: 400 });
  }

  const system = {
    role: 'system' as const,
    content:
      'You are a strict interviewer. Provide objective evaluation on a 0-10 scale. Be concise and specific. Return only JSON.',
  };
  const user = {
    role: 'user' as const,
    content: `Question: ${question}\nAnswer: ${candidateAnswer}\n\nContext CV (optional):\n${cvText || ''}\n\nContext JD (optional):\n${jdText || ''}\n\nReturn fields: score (0-10 integer), strengths[], improvements[], summary (<= 60 words), suggested_answer (concise model answer).`,
  };

  const data = await llmJson<Evaluation>([system, user], req, 0.2);

  return Response.json({ evaluation: data });
}