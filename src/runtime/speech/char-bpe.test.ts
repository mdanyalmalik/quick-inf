import { describe, expect, it } from 'vitest';
import { CharBpeTokenizer } from './char-bpe';

// A miniature of Soprano's tokenizer.json: added tokens, a few characters and merges.
const tokenizer = new CharBpeTokenizer({
  added_tokens: [
    { id: 0, content: '[UNK]' },
    { id: 1, content: '[TEXT]' },
    { id: 2, content: '[START]' },
  ],
  model: {
    type: 'BPE',
    unk_token: '[UNK]',
    vocab: { '[UNK]': 0, '[TEXT]': 1, '[START]': 2, h: 3, e: 4, l: 5, o: 6, ' ': 7, he: 8, ll: 9, hell: 10, '4': 11, '2': 12, '.': 13 },
    merges: [
      ['h', 'e'],
      ['l', 'l'],
      ['he', 'll'],
    ],
  },
});

describe('CharBpeTokenizer', () => {
  it('matches added tokens first, merges by rank, splits digits and maps unknown characters to [UNK]', () => {
    expect(tokenizer.encode('[TEXT]hello 42.[START]')).toEqual([1, 10, 6, 7, 11, 12, 13, 2]);
    expect(tokenizer.encode('hex')).toEqual([8, 0]);
  });
});
