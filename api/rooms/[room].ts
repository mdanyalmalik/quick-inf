// The pipeline's signalling relay on Vercel (see ../_relay.ts), storing rooms in the project's Upstash Redis.

import { handleRelay, restStore } from '../_relay.js';

const store = () => {
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) throw new Error('The relay has no database (KV_REST_API_URL, KV_REST_API_TOKEN)');
  return restStore(url, token);
};

async function respond(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const room = url.pathname.split('/').pop() ?? '';
  const client = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 'unknown';
  try {
    const result = await handleRelay(
      { method: request.method, room, after: Number(url.searchParams.get('after') ?? 0), body: request.method === 'POST' ? await request.text() : null, client },
      store(),
    );
    return Response.json(result.body, { status: result.status, headers: { 'cache-control': 'no-store' } });
  } catch (err) {
    console.error(err);
    return Response.json({ error: 'relay unavailable' }, { status: 503 });
  }
}

export const GET = respond;
export const POST = respond;
