import { describe, expect, it } from 'vitest';
import { numberToWords, ordinalToWords, pairsToWords } from './english-numbers';

// Expected values are Python inflect's (number_to_words with andword='').
describe('numberToWords', () => {
  it.each([
    ['0', 'zero'],
    ['7', 'seven'],
    ['15', 'fifteen'],
    ['21', 'twenty-one'],
    ['100', 'one hundred'],
    ['101', 'one hundred one'],
    ['123', 'one hundred twenty-three'],
    ['1001', 'one thousand one'],
    ['1234', 'one thousand, two hundred thirty-four'],
    ['12345', 'twelve thousand, three hundred forty-five'],
    ['100000', 'one hundred thousand'],
    ['1234567', 'one million, two hundred thirty-four thousand, five hundred sixty-seven'],
    ['1000000000', 'one billion'],
    ['0142', 'one hundred forty-two'],
  ])('%s', (digits, words) => expect(numberToWords(digits)).toBe(words));
});

describe('ordinalToWords', () => {
  it.each([
    ['1st', 'first'],
    ['2nd', 'second'],
    ['3rd', 'third'],
    ['4th', 'fourth'],
    ['11th', 'eleventh'],
    ['12th', 'twelfth'],
    ['20th', 'twentieth'],
    ['21st', 'twenty-first'],
    ['100th', 'one hundredth'],
    ['123rd', 'one hundred twenty-third'],
  ])('%s', (digits, words) => expect(ordinalToWords(digits)).toBe(words));
});

describe('pairsToWords', () => {
  it.each([
    ['1990', 'nineteen, ninety'],
    ['1905', 'nineteen, oh five'],
    ['1024', 'ten, twenty-four'],
    ['1100', 'eleven, oh oh'],
    ['1001', 'ten, oh one'],
  ])('%s', (digits, words) => expect(pairsToWords(digits)).toBe(words));
});
