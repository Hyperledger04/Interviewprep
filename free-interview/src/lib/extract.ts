export async function extractFromPdf(buffer: Buffer): Promise<string> {
  const pdf = (await import('pdf-parse')).default as (b: Buffer) => Promise<{ text: string }>;
  const data = await pdf(buffer);
  return data.text || '';
}

export async function extractFromDocx(buffer: Buffer): Promise<string> {
  const mammoth = await import('mammoth');
  const { value } = await mammoth.extractRawText({ buffer });
  return value || '';
}

export function splitIntoChunks(text: string, chunkSize = 1800): string[] {
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + chunkSize));
    start += chunkSize;
  }
  return chunks;
}