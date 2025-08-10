import { NextRequest, NextResponse } from 'next/server';

export const config = {
  matcher: ['/api/:path*'],
};

export function middleware(req: NextRequest) {
  const res = NextResponse.next();
  try {
    const clientKey = req.headers.get('x-api-key') || req.cookies.get('api_key')?.value;
    if (clientKey) {
      res.headers.set('x-api-key', clientKey);
    }
  } catch {}
  return res;
}