import { describe, expect, it } from 'vitest';
import { inflectChunks, inflectNormalize } from './inflect-text';
import { kittenChunks, kittenNormalize } from './kitten-text';

// Expected values are the Python packages' (kittentts TextPreprocessor, Inflect's normalize_text).
describe('kittenNormalize', () => {
  it.each([
    ['There are 1200 students and 42 teachers.', 'there are twelve hundred students and forty two teachers'],
    ['A coffee costs $4.99 here.', 'a coffee costs four dollars and ninety nine cents here'],
    ['Inflation rose by 3.5% last quarter.', 'inflation rose by three point five percent last quarter'],
    ['She finished 1st, he came 2nd, I was 3rd.', 'she finished first he came second i was third'],
    ['Add 3/4 cup of sugar.', 'add three quarters cup of sugar'],
    ['The 80s music scene was iconic.', 'the eighties music scene was iconic'],
    ['Call us at 555-1234 or 1-800-555-0199.', 'call us at five five five one two three four or one eight zero zero five five five zero one nine nine'],
    ["I don't know, won't you help? They've already left.", 'i do not know will not you help they have already left'],
    ['Learning rate is 1e-4.', 'learning rate is one times ten to the negative four'],
    ['We trained a 7B parameter model.', 'we trained a seven billion parameter model'],
    ['Download the 2.5GB model file.', 'download the two point five gigabytes model file'],
    ['Visit https://example.com or email hello@example.com.', 'visit or email'],
    ['Read pages 10-20 for homework.', 'read pages ten to twenty for homework'],
    ['gpt-3 is pretty sick.', 'gpt three is pretty sick'],
    ['Temperature dropped to -5 degrees.', 'temperature dropped to negative five degrees'],
    ['Rent is €1,200 per month.', 'rent is twelve hundred euros per month'],
    ['Connect to 192.168.1.1 now.', 'connect to one nine two dot one six eight dot one dot one now'],
    ['Our meeting moved to March 14th, 2027, in room 302.', 'our meeting moved to march fourteenth two thousand twenty seven in room three hundred two'],
  ])('%s', (text, out) => expect(kittenNormalize(text)).toBe(out));

  it('keeps the space after a time and reads tens ordinals properly (two fixes over the package)', () => {
    expect(kittenNormalize('Open at 9:30 and on the 20th.')).toBe('open at nine thirty and on the twentieth');
  });

  it('chunks end in punctuation', () => {
    expect(kittenChunks('hello there')).toEqual(['hello there,']);
    expect(kittenChunks(`${'word '.repeat(100)}end`).every((c) => c.length <= 401 && c.endsWith(','))).toBe(true);
  });
});

describe('inflectNormalize', () => {
  it.each([
    ['Dr. Smith lives at 221 North Main St. in apt 4B.', 'doctor Smith lives at two two one North Main saint in apt four bee.'],
    ['It costs $1,234.50 or $1.', 'It costs one thousand two hundred and thirty four dollars and fifty cents or one dollar.'],
    ['Meet me on 12/25/2024 at 3:05 p.m. sharp.', 'Meet me on December twenty fifth two thousand and twenty four at three oh five p m. sharp.'],
    ['The U.S.A. and NASA use RTX 4090 GPUs.', 'The U S A and en ay ess ay use ar tee ex forty ninety GPUs.'],
    ['Call 555-0142 now; version 1.2.3 is out (finally).', 'Call five five five, zero one four two now; version one point two point three is out, finally.'],
    ['Pi is 3.14 and 12345 is big but 2024 is a year; 7th place.', 'Pi is three point one four and one two three four five is big but two thousand and twenty four is a year; seventh place.'],
    ['He said “hello” — then left…', 'He said "hello", then left. . .'],
    ['Our meeting moved to March 14th, 2027, in room 302.', 'Our meeting moved to March fourteenth, two thousand and twenty seven, in room three oh two.'],
  ])('%s', (text, out) => expect(inflectNormalize(text)).toBe(out));

  it('chunks at sentence ends', () => {
    expect(inflectChunks('Hello from Phonon.  This is a quick test.')).toEqual(['Hello from Phonon.', 'This is a quick test.']);
  });
});
