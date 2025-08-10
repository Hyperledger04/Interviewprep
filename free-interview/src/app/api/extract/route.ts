import { NextRequest } from 'next/server';
import { extractFromDocx, extractFromPdf } from '@/lib/extract';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const files = form.getAll('files');
  if (!files?.length) {
    return new Response(JSON.stringify({ error: 'No files provided' }), { status: 400 });
  }

  const texts: string[] = [];

  for (const file of files) {
    if (!(file instanceof File)) continue;
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const name = file.name.toLowerCase();

    if (name.endsWith('.pdf')) {
      const text = await extractFromPdf(buffer);
      texts.push(text);
    } else if (name.endsWith('.docx')) {
      const text = await extractFromDocx(buffer);
      texts.push(text);
    } else if (name.endsWith('.txt')) {
      texts.push(buffer.toString('utf8'));
    } else {
      texts.push('');
    }
  }

  return Response.json({ text: texts.filter(Boolean).join('\n\n') });
}