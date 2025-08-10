import { NextRequest } from 'next/server';
import { llmJson } from '@/lib/llm';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export type GeneratedQuestion = {
  id: string;
  question: string;
  category: string;
  difficulty: 'easy' | 'medium' | 'hard';
};

export async function POST(req: NextRequest) {
  const { cvText, jdText, numQuestions = 10 } = await req.json();
  if (!cvText && !jdText) {
    return new Response(JSON.stringify({ error: 'cvText or jdText required' }), { status: 400 });
  }

  const system = {
    role: 'system' as const,
    content:
      'You are a senior interviewer. Generate concise, specific interview questions strictly grounded in the candidate CV and the job description. Avoid duplicates. Return only JSON.',
  };
  const user = {
    role: 'user' as const,
    content: `CV:\n${cvText || ''}\n\nJD:\n${jdText || ''}\n\nReturn a JSON object with an array questions sized ${numQuestions} with fields: id, question, category, difficulty (easy|medium|hard).`,
  };

  const data = await llmJson<{ questions: GeneratedQuestion[] }>([system, user], req);

  return Response.json({ questions: data.questions || [] });
}