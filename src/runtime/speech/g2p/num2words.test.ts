import { describe, expect, it } from 'vitest';
import { cardinal, decimal, ordinal, year } from './num2words';

// Expected values are Python num2words' (lang EN).
describe('num2words', () => {
  it.each([
    ['0', 'zero'],
    ['21', 'twenty-one'],
    ['101', 'one hundred and one'],
    ['142', 'one hundred and forty-two'],
    ['1001', 'one thousand and one'],
    ['1100', 'one thousand, one hundred'],
    ['1234', 'one thousand, two hundred and thirty-four'],
    ['2027', 'two thousand and twenty-seven'],
    ['100001', 'one hundred thousand and one'],
    ['1234567', 'one million, two hundred and thirty-four thousand, five hundred and sixty-seven'],
    ['1000000000', 'one billion'],
    ['-5', 'minus five'],
  ])('cardinal %s', (n, words) => expect(cardinal(n)).toBe(words));

  it.each([
    ['1', 'first'],
    ['12', 'twelfth'],
    ['20', 'twentieth'],
    ['21', 'twenty-first'],
    ['142', 'one hundred and forty-second'],
  ])('ordinal %s', (n, words) => expect(ordinal(n)).toBe(words));

  it.each([
    ['142', 'one forty-two'],
    ['1900', 'nineteen hundred'],
    ['1905', 'nineteen oh-five'],
    ['2000', 'two thousand'],
    ['2005', 'two thousand and five'],
    ['2010', 'twenty ten'],
    ['2027', 'twenty twenty-seven'],
  ])('year %s', (n, words) => expect(year(n)).toBe(words));

  it.each([
    ['3.14', 'three point one four'],
    ['10.01', 'ten point zero one'],
    ['1.0', 'one'],
    ['2.50', 'two point five'],
  ])('decimal %s', (n, words) => expect(decimal(n)).toBe(words));
});
