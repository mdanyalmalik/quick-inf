// GPT-2 style byte <-> unicode mapping used by byte-level BPE vocabularies.
// Printable bytes map to themselves; the rest are shifted to code points >= 256
// so every byte has a visible, non-whitespace stand-in (e.g. space -> "Ġ").

function buildByteToChar(): string[] {
  const printable: number[] = [];
  for (let b = 0x21; b <= 0x7e; b++) printable.push(b);
  for (let b = 0xa1; b <= 0xac; b++) printable.push(b);
  for (let b = 0xae; b <= 0xff; b++) printable.push(b);

  const table = new Array<string>(256);
  for (const b of printable) table[b] = String.fromCharCode(b);
  let shifted = 0;
  for (let b = 0; b < 256; b++) {
    if (table[b] === undefined) table[b] = String.fromCharCode(256 + shifted++);
  }
  return table;
}

export const BYTE_TO_CHAR: readonly string[] = buildByteToChar();

export const CHAR_TO_BYTE: ReadonlyMap<string, number> = new Map(BYTE_TO_CHAR.map((ch, b) => [ch, b]));

const utf8Encoder = new TextEncoder();

/** UTF-8 encodes `text` and maps each byte to its stand-in character. */
export function textToByteChars(text: string): string {
  let out = '';
  for (const b of utf8Encoder.encode(text)) out += BYTE_TO_CHAR[b];
  return out;
}

/** Reverses `textToByteChars` for a token string, appending raw bytes to `out`. */
export function byteCharsToBytes(token: string, out: number[]): void {
  for (const ch of token) {
    const b = CHAR_TO_BYTE.get(ch);
    if (b === undefined) throw new Error(`Character ${JSON.stringify(ch)} is not in the byte-level alphabet`);
    out.push(b);
  }
}
