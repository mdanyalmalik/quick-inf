// Chatterbox's text handling (resemble-ai/chatterbox, MIT): punc_norm, the clean-up its Python package runs on each
// text before tokenizing, and tokenizing with its tags ("[laugh]", "[sigh]": added tokens matched as written). Tests
// compare with the Python and Hugging Face's tokenizers (chatterbox-text.test.ts).

/**
 * punc_norm: the first letter capitalised, whitespace runs collapsed, uncommon punctuation replaced (Turbo's list is
 * shorter than the original model's), and a full stop added unless the text ends in . ! ? - or , (or, for the
 * multilingual model, a Chinese or Japanese sentence end).
 */
export function puncNorm(text: string, variant: 'turbo' | 'original' | 'multilingual'): string {
  if (text.length === 0) return 'You need to add some text for me to talk.';
  const first = String.fromCodePoint(text.codePointAt(0)!);
  if (/\p{Ll}/u.test(first)) text = first.toUpperCase() + text.slice(first.length);
  text = text.split(/\s+/).filter(Boolean).join(' ');
  for (const [from, to] of variant === 'turbo' ? TURBO_PUNCTUATION : ORIGINAL_PUNCTUATION) text = text.replaceAll(from, to);
  text = text.replace(/ +$/, '');
  const enders = variant === 'multilingual' ? ['.', '!', '?', '-', ',', '、', '，', '。', '？', '！'] : ['.', '!', '?', '-', ','];
  if (!enders.some((p) => text.endsWith(p))) text += '.';
  return text;
}

const TURBO_PUNCTUATION: [string, string][] = [
  ['…', ', '],
  [':', ','],
  ['—', '-'],
  ['–', '-'],
  [' ,', ','],
  ['“', '"'],
  ['”', '"'],
  ['‘', "'"],
  ['’', "'"],
];

const ORIGINAL_PUNCTUATION: [string, string][] = [
  ['...', ', '],
  ['…', ', '],
  [':', ','],
  [' - ', ', '],
  [';', ', '],
  ['—', '-'],
  ['–', '-'],
  [' ,', ','],
  ['“', '"'],
  ['”', '"'],
  ['‘', "'"],
  ['’', "'"],
];

/**
 * Chatterbox Multilingual's text for a language (the model card's prepare_language): Korean syllables decomposed
 * into jamo, then the language tag in front ("[fr]…"). Chinese, Japanese and Hebrew need converters this doesn't have.
 */
export function prepareLanguage(text: string, language: string): string {
  if (language === 'ko') {
    text = Array.from(text, (c) => {
      const base = c.charCodeAt(0) - 0xac00;
      if (base < 0 || c > '힯') return c;
      const final = base % 28;
      return (
        String.fromCharCode(0x1100 + Math.floor(base / (21 * 28)), 0x1161 + Math.floor((base % (21 * 28)) / 28)) +
        (final ? String.fromCharCode(0x11a7 + final) : '')
      );
    })
      .join('')
      .trim();
  }
  return `[${language.toLowerCase()}]${text}`;
}

/**
 * Token ids of `text` with the tokenizer's added tokens (`tags`, e.g. "[chuckle]") picked out wherever they're
 * written, as Hugging Face's tokenizers split them, and the text between them encoded.
 */
export function encodeWithTags(
  tokenizer: { encode(text: string): number[]; specialTokenId(content: string): number },
  text: string,
  tags: readonly string[],
): number[] {
  if (tags.length === 0) return tokenizer.encode(text);
  const pattern = new RegExp([...tags].sort((a, b) => b.length - a.length).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');
  const ids: number[] = [];
  let last = 0;
  for (const m of text.matchAll(pattern)) {
    if (m.index > last) ids.push(...tokenizer.encode(text.slice(last, m.index)));
    ids.push(tokenizer.specialTokenId(m[0]));
    last = m.index + m[0].length;
  }
  if (last < text.length) ids.push(...tokenizer.encode(text.slice(last)));
  return ids;
}
