import { describe, expect, it } from 'vitest';
import { cleanText, splitSentences } from './soprano-text';

// Expected values are the Python original's (soprano/utils/text_normalizer.py clean_text).
describe('cleanText', () => {
  it.each([
    ['1,2,3,456,176', 'one hundred twenty-three million, four hundred fifty-six thousand, one hundred seventy-six.'],
    ['123-456-7890', 'one two three, four five six, seven eight nine zero.'],
    ['(111) 111-1111', 'one one one, one one one, one one one one.'],
    ['$2.47', 'two dollars, forty-seven cents.'],
    ['$247', 'two hundred forty-seven dollars.'],
    ['$0.27', 'twenty-seven cents.'],
    ['$1.00', 'one dollar.'],
    ['1990', 'nineteen ninety.'],
    ['2005', 'two thousand five.'],
    ['2010', 'twenty ten.'],
    ['1024', 'ten twenty-four.'],
    ['2.47023', 'two point four seven zero two three.'],
    ['1/1/2025', 'one dash one dash twenty twenty-five.'],
    ['A 1/1 A', 'a one over one a.'],
    ['8:00', "eight o'clock."],
    ['8:05', 'eight oh five.'],
    ['13:00', 'thirteen minutes.'],
    ['-1 + 2 * 3 - 4 / 5', 'minus one plus two times three minus four over five.'],
    ['100k', 'one hundred thousand.'],
    ['#1', 'number one.'],
    ['1st 2nd 3rd 4th', 'first second third fourth.'],
    ['and/or', 'and or.'],
    ['LMDeployDecoderModel', 'l m deploy decoder model.'],
    ['UPPERCASE', 'uppercase.'],
    ['TPUs', "tpu's."],
    ['Dr. Smith & Mr. Jones met at 5 pm.', 'doctor smith and mister jones met at five pm.'],
    ['Wait... what?!', 'wait... what?'],
    ['Café — naïve résumé', 'cafe, naive resume.'],
    ['(see above) then go', ', see above, then go.'],
    ['e.g. a.b', 'e dot g. a dot b.'],
    ['https://x.com', 'h t t p s colon slash slash x dot com.'],
    ['50% off <3 cool', 'fifty percent off less than three cool.'],
    ['Please call me back at 555 0142 before 6 pm on Friday.', 'please call me back at five hundred fifty-five one hundred forty-two before six pm on friday.'],
    ['Our meeting moved to March 14th, 2027, in room 302.', 'our meeting moved to march fourteenth, twenty twenty-seven, in room three hundred two.'],
    ['Send the report to anna.lee@example.com by the end of the week.', 'send the report to anna dot lee at example dot com by the end of the week.'],
  ])('%s', (input, output) => expect(cleanText(input)).toBe(output));
});

describe('splitSentences', () => {
  it('splits at sentence ends and merges short sentences into their neighbour', () => {
    expect(splitSentences('hello there. this is a much longer sentence that stands alone. ok!')).toEqual([
      'hello there. this is a much longer sentence that stands alone. ok!',
    ]);
    expect(splitSentences('this first sentence is long enough on its own. and so is this second one, really.')).toEqual([
      'this first sentence is long enough on its own.',
      'and so is this second one, really.',
    ]);
  });
});
