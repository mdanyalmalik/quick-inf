import { describe, expect, it } from 'vitest';
import { chunkText, prepareText, textIds } from './supertonic-text';

describe('prepareText', () => {
  it('normalises quotes and dashes, fixes spacing and wraps the text in its language tag', () => {
    expect(prepareText('“Hello” — world , again', 'en')).toBe('<en>"Hello" - world, again.</en>');
  });

  it('keeps ending punctuation, spells out @ and drops emoji', () => {
    expect(prepareText('Mail me @ home 😀!', 'en')).toBe('<en>Mail me at home!</en>');
  });
});

describe('chunkText', () => {
  it('joins sentences up to the length limit and splits paragraphs', () => {
    expect(chunkText('One. Two? Three!\n\nFour.', 10)).toEqual(['One. Two?', 'Three!', 'Four.']);
  });

  it("doesn't break after a title", () => {
    expect(chunkText('Dr. Smith is here. Hi.', 18)).toEqual(['Dr. Smith is here.', 'Hi.']);
  });
});

describe('textIds', () => {
  it('maps characters through the index, unknown ones to 0', () => {
    const indexer = new Array(128).fill(-1);
    indexer['a'.charCodeAt(0)] = 5;
    expect(Array.from(textIds('ab', indexer), Number)).toEqual([5, 0]);
  });
});
