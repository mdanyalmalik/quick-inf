// Size of stored records, measured as their UTF-8 JSON encoding. Browsers don't
// report IndexedDB's on-disk bytes per store, so this measures the data itself
// (close to, but not exactly, what the database uses on disk).

const encoder = new TextEncoder();

export function dataBytes(records: unknown[]): number {
  let total = 0;
  for (const r of records) total += encoder.encode(JSON.stringify(r)).length;
  return total;
}
