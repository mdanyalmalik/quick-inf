// VoxCPM2's text ids: its Llama-style SentencePiece BPE (no BOS), with every vocabulary token made only of CJK
// ideographs (U+4E00-U+9FFF, a "▁" aside) split into its single characters, as the voxcpm package's
// mask_multichar_chinese_tokens does (the model was trained on single characters).

import type { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';

/** Splits of the multi-character Chinese tokens, by id, from the tokenizer.json's vocabulary. */
export function voxcpmSplits(tokenizerJson: { model: { vocab: Record<string, number>; unk_token?: string | null } }): Map<number, number[]> {
  const vocab = tokenizerJson.model.vocab;
  const unk = vocab[tokenizerJson.model.unk_token ?? '<unk>'] ?? 0;
  const splits = new Map<number, number[]>();
  for (const [token, id] of Object.entries(vocab)) {
    const chars = [...token.replaceAll('▁', '')];
    if (chars.length >= 2 && chars.every((c) => c >= '一' && c <= '鿿')) splits.set(id, chars.map((c) => vocab[c] ?? unk));
  }
  return splits;
}

export function voxcpmIds(tokenizer: BpeTokenizer, splits: Map<number, number[]>, text: string): number[] {
  return tokenizer.encode(text).flatMap((id) => splits.get(id) ?? [id]);
}
