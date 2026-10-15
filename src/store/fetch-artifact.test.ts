import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArtifactFile } from '../catalogue/models';
import { ArtifactError, fetchArtifact, retryDelayMs, STALL_MS, streamArtifact, verifySha256 } from './fetch-artifact';

const FILE: ArtifactFile = {
  path: 'model.bin',
  sizeBytes: 8,
  // sha256 of bytes 0..7
  sha256: '8a851ff82ee7048ad09ec3847f1ddf44944104d2cbd17ef4e3db22c6785a0d45',
};

/** A response whose body yields the given chunks, then optionally fails or stalls. */
function streamResponse(
  chunks: number[][],
  end: 'close' | 'error' | 'stall' = 'close',
  status = 200,
  signal?: AbortSignal,
  headers?: Record<string, string>,
): Response {
  let i = 0;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      // Like real fetch, aborting errors the body.
      signal?.addEventListener('abort', () => controller.error(new DOMException('aborted', 'AbortError')));
    },
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(new Uint8Array(chunks[i++]!));
      } else if (end === 'close') {
        controller.close();
      } else if (end === 'error') {
        controller.error(new TypeError('network error'));
      } else {
        return new Promise(() => {});
      }
    },
  });
  return new Response(body, { status, headers });
}

function mockFetch(response: Response | Error) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init?: RequestInit) => {
      if (response instanceof Error) throw response;
      init?.signal?.throwIfAborted();
      return response;
    }),
  );
}

async function codeOf(promise: Promise<unknown>): Promise<string> {
  try {
    await promise;
    return 'resolved';
  } catch (err) {
    return err instanceof ArtifactError ? err.code : `other: ${String(err)}`;
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fetchArtifact', () => {
  it('returns the bytes and reports progress', async () => {
    mockFetch(streamResponse([[0, 1, 2], [3, 4, 5, 6, 7]]));
    const progress: number[] = [];
    const bytes = await fetchArtifact('u', FILE, (p) => progress.push(p.loadedBytes));
    expect([...bytes]).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(progress).toEqual([3, 8]);
    await expect(verifySha256(bytes, FILE)).resolves.toBeUndefined();
  });

  it('reports an HTTP error as download-failed', async () => {
    mockFetch(streamResponse([], 'close', 404));
    expect(await codeOf(fetchArtifact('u', FILE, () => {}))).toBe('download-failed');
  });

  it('reports a failed request as download-failed', async () => {
    mockFetch(new TypeError('Failed to fetch'));
    expect(await codeOf(fetchArtifact('u', FILE, () => {}))).toBe('download-failed');
  });

  it('reports a connection dropped mid-download as download-failed', async () => {
    mockFetch(streamResponse([[0, 1, 2]], 'error'));
    expect(await codeOf(fetchArtifact('u', FILE, () => {}))).toBe('download-failed');
  });

  it('reports a truncated file as integrity-mismatch', async () => {
    mockFetch(streamResponse([[0, 1, 2]]));
    expect(await codeOf(fetchArtifact('u', FILE, () => {}))).toBe('integrity-mismatch');
  });

  it('reports an oversized file as integrity-mismatch', async () => {
    mockFetch(streamResponse([[0, 1, 2, 3, 4], [5, 6, 7, 8]]));
    expect(await codeOf(fetchArtifact('u', FILE, () => {}))).toBe('integrity-mismatch');
  });

  it('reports altered bytes as integrity-mismatch', async () => {
    const bytes = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 9]);
    expect(await codeOf(verifySha256(bytes, FILE))).toBe('integrity-mismatch');
  });

  it('reports cancel before the response as cancelled', async () => {
    mockFetch(streamResponse([[0]]));
    const abort = new AbortController();
    abort.abort();
    expect(await codeOf(fetchArtifact('u', FILE, () => {}, abort.signal))).toBe('cancelled');
  });

  it('reports cancel mid-download as cancelled and stops reading', async () => {
    const abort = new AbortController();
    mockFetch(streamResponse([[0, 1, 2]], 'stall', 200, abort.signal));
    const result = codeOf(
      fetchArtifact('u', FILE, (p) => {
        if (p.loadedBytes === 3) setTimeout(() => abort.abort());
      }, abort.signal),
    );
    expect(await result).toBe('cancelled');
  });
});

describe('streamArtifact resume', () => {
  /** Runs a resumed download from `offset` over a buffer pre-filled with 0..offset-1. */
  async function resume(response: Response, offset: number) {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => response);
    vi.stubGlobal('fetch', fetchMock);
    const bytes = new Uint8Array(FILE.sizeBytes);
    bytes.set([0, 1, 2, 3, 4, 5, 6, 7].slice(0, offset));
    const writes: number[] = [];
    let restarted = false;
    const transferred = await streamArtifact('u', FILE, {
      offset,
      write: (chunk, at) => {
        writes.push(at);
        bytes.set(chunk, at);
      },
      onRestart: () => {
        restarted = true;
        bytes.fill(0);
      },
    });
    const headers = new Headers(fetchMock.mock.calls[0]?.[1]?.headers);
    return { bytes: [...bytes], writes, restarted, transferred, range: headers.get('range') };
  }

  it('asks for the rest and continues at the offset on a matching 206', async () => {
    const r = await resume(streamResponse([[5, 6, 7]], 'close', 206, undefined, { 'Content-Range': 'bytes 5-7/8' }), 5);
    expect(r.range).toBe('bytes=5-');
    expect(r.restarted).toBe(false);
    expect(r.writes).toEqual([5]);
    expect(r.transferred).toBe(3);
    expect(r.bytes).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it('starts over when the server ignores the Range request (200)', async () => {
    const r = await resume(streamResponse([[0, 1, 2, 3], [4, 5, 6, 7]]), 5);
    expect(r.restarted).toBe(true);
    expect(r.writes).toEqual([0, 4]);
    expect(r.transferred).toBe(8);
    expect(r.bytes).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it('starts over when the 206 range starts somewhere else', async () => {
    const r = await resume(
      streamResponse([[0, 1, 2, 3, 4, 5, 6, 7]], 'close', 206, undefined, { 'Content-Range': 'bytes 0-7/8' }),
      5,
    );
    expect(r.restarted).toBe(true);
    expect(r.writes).toEqual([0]);
  });

  it('does nothing when the file is already complete', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    expect(await streamArtifact('u', FILE, { offset: 8, write: () => {} })).toBe(0);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('passes errors from the writer through unchanged (e.g. disk full)', async () => {
    mockFetch(streamResponse([[0, 1, 2]]));
    const diskFull = new DOMException('full', 'QuotaExceededError');
    const err = await streamArtifact('u', FILE, {
      write: () => {
        throw diskFull;
      },
    }).catch((e: unknown) => e);
    expect(err).toBe(diskFull);
  });
});

describe('download failure classes', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('treats a missing or forbidden file as permanent, and server trouble as worth retrying', async () => {
    for (const [status, transient] of [
      [404, false],
      [403, false],
      [503, true],
      [429, true],
    ] as const) {
      vi.stubGlobal('fetch', async () => new Response(null, { status }));
      const err = await fetchArtifact('u', FILE, () => {}).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(ArtifactError);
      expect((err as ArtifactError).transient).toBe(transient);
    }
  });

  it('treats a dropped connection as worth retrying', async () => {
    vi.stubGlobal('fetch', async () => streamResponse([[0, 1]], 'error'));
    const err = (await fetchArtifact('u', FILE, () => {}).catch((e: unknown) => e)) as ArtifactError;
    expect(err.code).toBe('download-failed');
    expect(err.transient).toBe(true);
  });

  it('gives up on a stalled transfer, as a retryable failure', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', async (_: string, init: RequestInit) => streamResponse([[0, 1]], 'stall', 200, init.signal ?? undefined));
    const pending = fetchArtifact('u', FILE, () => {}).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(STALL_MS + 1);
    const err = (await pending) as ArtifactError;
    expect(err.message).toMatch(/stalled/);
    expect(err.transient).toBe(true);
    vi.useRealTimers();
  });

  it('waits longer before each retry, with jitter, up to 20 s', () => {
    expect(retryDelayMs(1, () => 0.5)).toBe(1000);
    expect(retryDelayMs(3, () => 0.5)).toBe(4000);
    expect(retryDelayMs(10, () => 0.5)).toBe(20000);
    expect(retryDelayMs(1, () => 0)).toBe(500);
  });
});
