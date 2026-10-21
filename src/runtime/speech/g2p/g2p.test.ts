import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { EnglishG2P, INFLECT_SYMBOLS, KITTEN_SYMBOLS, loadLexicon, parseLexicon, wordToIpa } from './index';

const LEXICON = new URL('../../../../exports/speech/g2p/en-us-lexicon.txt.gz', import.meta.url);

describe('parseLexicon', () => {
  it('reads front-coded words, ascii phoneme stand-ins and heteronyms, and grows capitalised copies', () => {
    const text = '# test\n!symbols 5] əˈ\n@gold\n0record\tr]6kord|VERB=r5k]ord\n2d\td]5d\n@silver\n0zed\tz]6d\n';
    const lex = parseLexicon(text);
    expect(lex.golds.get('record')).toEqual({ DEFAULT: 'rˈ6kord', VERB: 'rəkˈord' });
    expect(lex.golds.get('red')).toBe('dˈəd');
    expect(lex.golds.get('Red')).toBe('dˈəd');
    expect(lex.silvers.get('zed')).toBe('zˈ6d');
  });
});

describe('EnglishG2P', () => {
  let g: EnglishG2P;
  beforeAll(async () => {
    g = new EnglishG2P(await loadLexicon(readFileSync(LEXICON)));
  });

  // Expected values are Python misaki's (en.G2P, trf=False, American English).
  it.each([
    ['Hello from Phonon. This is a quick test.', 'həlˈO fɹʌm fˈOnˌɑn. ðˌɪs ɪz ɐ kwˈɪk tˈɛst.'],
    ['The quick brown fox jumps over the lazy dog near the riverbank.', 'ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ nˌɪɹ ðə ɹˈɪvəɹbˌæŋk.'],
    ['Please call me back at 555 0142 before 6 pm on Friday.', 'plˈiz kˈɔl mˌi bˈæk æt fˈIv hˈʌndɹəd fˈɪfti fˈIv wˈʌn fˈɔɹTi tˈu bəfˈɔɹ sˈɪks pˌiˈɛm ˌɔn fɹˈIdˌA.'],
    ['Our meeting moved to March 14th, 2027, in room 302.', 'ˌWəɹ mˈiTɪŋ mˈuvd tə mˈɑɹʧ fˌɔɹtˈinθ, twˈɛnti twˈɛnti sˈɛvən, ɪn ɹˈum θɹˈi hˈʌndɹəd tˈu.'],
    ['Photosynthesis turns sunlight, water and carbon dioxide into sugar and oxygen.', 'fˌOTəsˈɪnθəsɪs tˈɜɹnz sˈʌnlˌIt, wˈɔTəɹ ænd kˈɑɹbən dIˈɑksˌId ˈɪntu ʃˈʊɡəɹ ænd ˈɑksəʤᵊn.'],
    ['I would like two coffees and a blueberry muffin, please.', 'ˌI wʊd lˈIk tˈu kˈɔfiz ænd ɐ blˈubˌɛɹi mˈʌfᵊn, plˈiz.'],
    ['After the storm passed, the streets were quiet and the air smelled of rain.', 'ˈæftəɹ ðə stˈɔɹm pˈæst, ðə stɹˈits wɜɹ kwˈIət ænd ði ˈɛɹ smˈɛld ʌv ɹˈAn.'],
    ["I can't believe it's already October; the leaves are turning gold.", 'ˌI kˈænt bəlˈiv ɪts ˌɔlɹˈɛdi ɑktˈObəɹ; ðə lˈivz ɑɹ tˈɜɹnɪŋ ɡˈOld.'],
    ['We\'ll need $12.50 for the tickets and another $3 for parking.', 'wˌil nˈid twˈɛlv dˈɑləɹz ænd fˈɪfti sˈɛnts fɔɹ ðə tˈɪkəts ænd ənˈʌðəɹ θɹˈi dˈɑləɹz fɔɹ pˈɑɹkɪŋ.'],
    ['The U.S. economy grew by 2.5% in the third quarter of 2024.', 'ðə jˌuˈɛs ikˈɑnəmi ɡɹˈu bI tˈu pYnt fˈIv pəɹsˈɛnt ɪn ðə θˈɜɹd kwˈɔɹTəɹ ʌv twˈɛnti twˈɛnti fˈɔɹ.'],
    ['They record a new album every year, and the last record sold well.', 'ðˌA ɹəkˈɔɹd ɐ nˈu ˈælbəm ˈɛvəɹi jˈɪɹ, ænd ðə lˈæst ɹˈɛkəɹd sˈOld wˈɛl.'],
    ['We used to live near the ocean, but now we live in the city.', 'wˌi jˈust tə lˈɪv nˌɪɹ ði ˈOʃən, bˌʌt nˈW wi lˈɪv ɪn ðə sˈɪTi.'],
    ["The cats' toys were scattered all over the living room floor.", 'ðə kˈæts tˈYz wɜɹ skˈæTəɹd ˈɔl ˈOvəɹ ðə lˈɪvɪŋ ɹˈum flˈɔɹ.'],
    ['An honest man walked into the hotel at an hour past midnight.', 'ɐn ˈɑnəst mˈæn wˈɔkt ˈɪntu ðə hOtˈɛl æt ɐn ˈWəɹ pˈæst mˈɪdnˌIt.'],
    ['Our well-known neighbour painted the fence bright yellow last weekend.', 'ˌWəɹ wˌɛlnˈOn nˈAbəɹ pˈAntᵻd ðə fˈɛns bɹˈIt jˈɛlO lˈæst wˈikˌɛnd.'],
  ])('misaki: %s', (text, ps) => expect(g.misaki(text)).toBe(ps));

  it('rebuilds forms the file leaves out from their stems', () => {
    expect(g.misaki('drivers')).toBe('dɹˈIvəɹz');
    expect(g.misaki("drivers'")).toBe('dɹˈIvəɹz');
    expect(g.misaki('messed')).toBe('mˈɛst');
  });

  it('reads unknown words without eSpeak', () => {
    expect(g.misaki('the fbi and the cia')).toBe('ði ˌɛfbˌiˈI ænd ðə sˌiˌIˈA');
    expect(g.misaki('café')).toBe('kæfˈA');
    expect(g.misaki('ay eye')).toBe('ˈA ˈI');
    expect(g.misaki('glorptastic')).toMatch(/^ɡl.*ˈ/);
    expect(g.misaki('xkcd', { fallback: false, unk: '?' })).toBe('?');
  });

  it('renders eSpeak-style IPA', () => {
    expect(g.ipa('The quick brown fox jumps over the lazy dog.')).toBe('ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˈoʊvɚ ðə lˈeɪzi dˈɔɡ.');
    expect(g.ipa('the end')).toBe('ðɪ ˈɛnd');
  });

  it('builds Kitten TTS input: tokens re-joined with spaces, ids framed by 0', () => {
    const [chunk] = g.kitten('Hello from Phonon. This is a quick test.');
    expect(chunk!.text).toBe('hello from phonon this is a quick test,');
    expect(chunk!.phonemes).toBe('həlˈoʊ fɹʌm fˈoʊnˌɑːn ðɪs ɪz ɐ kwˈɪk tˈɛst ,');
    expect(chunk!.ids[0]).toBe(0);
    expect(chunk!.ids.at(-1)).toBe(0);
    expect(chunk!.ids.length).toBe([...chunk!.phonemes].length + 2);
  });

  it('builds Inflect input: one chunk per sentence, ids interleaved with blanks', () => {
    const chunks = g.inflect('Hello from Phonon. This is a quick test.');
    expect(chunks.map((c) => c.phonemes)).toEqual(['həlˈoʊ fɹʌm fˈoʊnˌɑːn.', 'ðˌɪs ɪz ɐ kwˈɪk tˈɛst.']);
    const ids = chunks[0]!.ids;
    expect(ids.length).toBe([...chunks[0]!.phonemes].length * 2 + 1);
    expect(ids.filter((_, i) => i % 2 === 0).every((id) => id === 0)).toBe(true);
    expect(ids[1]).toBe(INFLECT_SYMBOLS.get('h'));
  });

  it('builds Paradee input sentence by sentence', () => {
    const vocab: Record<string, number> = Object.fromEntries([...' .həlˈOfɹʌmnɑðˌɪszɐkwt'].map((c, i) => [c, i + 1]));
    const chunks = g.paradee('Hello from Phonon. This is a quick test.', vocab);
    expect(chunks.map((c) => c.phonemes)).toEqual(['həlˈO fɹʌm fˈOnˌɑn.', 'ðˌɪs ɪz ɐ kwˈɪk tˈɛst.']);
    expect(chunks[0]!.ids).toEqual([0, ...[...chunks[0]!.phonemes].map((c) => vocab[c]), 0]);
  });
});

describe('wordToIpa', () => {
  it.each([
    ['ʤˈʌmps', 'jumps', 'dʒˈʌmps'],
    ['ɹˈɪvəɹbˌæŋk', 'riverbank', 'ɹˈɪvɚbˌæŋk'],
    ['tˈɜɹnz', 'turns', 'tˈɜːnz'],
    ['kˈɔl', 'call', 'kˈɔːl'],
    ['ˌɔn', 'on', 'ˌɔn'],
    ['lˈAzi', 'lazy', 'lˈeɪzi'],
    ['plˈiz', 'please', 'plˈiːz'],
    ['əbˈWt', 'about', 'ɐbˈaʊt'],
    ['ɹətˈɜɹn', 'return', 'ɹᵻtˈɜːn'],
    ['hˈʌndɹəd', 'hundred', 'hˈʌndɹɪd'],
    ['ɪnklˈud', 'include', 'ɪŋklˈuːd'],
    ['bˈʌʔn', 'button', 'bˈʌʔn̩'],
    ['mˈiTɪŋ', 'meeting', 'mˈiːɾɪŋ'],
  ])('%s (%s) → %s', (misaki, spelling, ipa) => expect(wordToIpa(misaki, spelling)).toBe(ipa));

  it('uses only symbols both eSpeak-trained models know', () => {
    const ipa = wordToIpa('ˈAIOWYʤʧTᵊɐᵻʔəɹɜɹɔɹɑɹ', '');
    expect([...ipa].every((c) => KITTEN_SYMBOLS.has(c) && INFLECT_SYMBOLS.has(c))).toBe(true);
  });
});
