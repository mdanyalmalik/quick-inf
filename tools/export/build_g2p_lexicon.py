"""Builds the English (US) pronunciation lexicon the browser G2P downloads (src/runtime/speech/g2p/lexicon.ts reads it)
from Misaki's us_gold.json and us_silver.json, into exports/speech/g2p/en-us-lexicon.txt.gz with its licence notice.

Entries the runtime derives anyway are left out: a capitalised copy of a lower-case entry with the same phonemes
(Misaki's grow_dictionary makes it), and a word whose phonemes Misaki's -s / -ed / -ing rules rebuild exactly from a
stem that stays in the file. Misaki's own Lexicon class does the check, so the runtime gives the same phonemes.

File format (UTF-8 text, gzip):
    # comment lines
    !symbols <ascii> <ipa>      each phoneme character in the entries below is the ascii one at the same position
    @gold / @silver             starts a section
    <p><rest>\t<phonemes>       the word is the previous word's first p characters (p in base 36) followed by <rest>
Phonemes are a Misaki string, or for heteronyms "<default>|TAG=<phonemes>|..." (an empty value means none).

    tools/export/.cache/tts-venv/bin/python -I tools/export/build_g2p_lexicon.py [--data DIR] [--out DIR]
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IPA = 'æðŋɑɔəɛɜɡɪɹɾʃʊʌʒʔʤʧˈˌθᵊᵻ'
ASCII = '0123456789#$%&*+<>[]{}~^'
BASE36 = '0123456789abcdefghijklmnopqrstuvwxyz'
NOTICE = """English (US) pronunciation lexicon for quick-inf's browser G2P.

Built by tools/export/build_g2p_lexicon.py from us_gold.json and us_silver.json of Misaki
(https://github.com/hexgrad/misaki, version {version}), Copyright hexgrad, licensed under the
Apache License, Version 2.0 (the text follows). Changes: the entries are re-encoded in a compact
text format (phoneme characters recoded to ASCII, words front-coded), and entries Misaki's lookup
rules rebuild exactly (capitalised copies, regular -s / -ed / -ing forms) are left out.
No eSpeak code or data is included.

"""


def encode(value):
    if isinstance(value, dict):
        value = '|'.join([value['DEFAULT'] or ''] + [f'{k}={v or ""}' for k, v in value.items() if k != 'DEFAULT'])
    return value.translate(str.maketrans(IPA, ASCII))


def front_coded(entries):
    lines, prev = [], ''
    for word in sorted(entries):
        n = 0
        while n < min(len(prev), len(word), len(BASE36) - 1) and prev[n] == word[n]:
            n += 1
        lines.append(f'{BASE36[n]}{word[n:]}\t{encode(entries[word])}\n')
        prev = word
    return ''.join(lines)


def drop_case_copies(d):
    """A capitalised entry equal to its lower-case one: Lexicon.grow_dictionary recreates it."""
    return {k: v for k, v in d.items()
            if not (len(k) > 1 and k != k.lower() and k == k.lower().capitalize() and d.get(k.lower()) == v)}


def drop_derivable(golds, silvers):
    """Lower-case words that Misaki's -s / -ed / -ing rules rebuild from a kept stem, for every tag and stress."""
    from misaki import en
    lex = en.Lexicon(british=False)
    lex.golds, lex.silvers = en.Lexicon.grow_dictionary(golds), en.Lexicon.grow_dictionary(silvers)
    ctx = en.TokenContext()
    tags = (None, 'NN', 'NNP', 'VB', 'VBD', 'VBN', 'JJ')
    stresses = (None, 0.5, 2)
    dropped = {'gold': set(), 'silver': set()}
    candidates = [(w, t) for t, d in (('gold', golds), ('silver', silvers)) for w, v in d.items()
                  if isinstance(v, str) and w == w.lower() and w.isalpha() and len(w) > 3 and w[-1] in 'sdg']
    for word, tier in sorted(candidates, key=lambda c: (len(c[0]), c[0])):
        table = lex.golds if tier == 'gold' else lex.silvers
        if tier == 'silver' and word in lex.golds:
            continue
        cap = word.capitalize()
        want = [lex.get_word(w, t, s, ctx)[0] for w in (word, cap) for t in tags for s in stresses]
        saved = {k: table.pop(k) for k in (word, cap) if k in table}
        got = [lex.get_word(w, t, s, ctx)[0] for w in (word, cap) for t in tags for s in stresses]
        if got == want:
            dropped[tier].add(word)
        else:
            table.update(saved)
    return ({k: v for k, v in golds.items() if k not in dropped['gold']},
            {k: v for k, v in silvers.items() if k not in dropped['silver']})


def main():
    import misaki
    from misaki import data
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path(data.__file__).parent)
    ap.add_argument('--out', type=Path, default=ROOT / 'exports/speech/g2p')
    args = ap.parse_args()
    golds = json.loads((args.data / 'us_gold.json').read_text())
    silvers = json.loads((args.data / 'us_silver.json').read_text())
    silvers = {k: v for k, v in silvers.items() if k not in golds}  # gold wins in Misaki's lookup
    n = len(golds) + len(silvers)
    golds, silvers = drop_derivable(drop_case_copies(golds), drop_case_copies(silvers))
    # The ascii stand-ins never occur in Misaki phonemes (letters and IPA only) nor clash with the separators.
    assert len(IPA) == len(ASCII) and not set(ASCII) & set('\t\n|=' + IPA)
    version = getattr(misaki, '__version__', '0.9.4')
    text = (f'# Misaki {version} English (US) lexicon (Apache-2.0); see LICENSE-misaki.txt\n'
            f'!symbols {ASCII} {IPA}\n@gold\n{front_coded(golds)}@silver\n{front_coded(silvers)}')
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'en-us-lexicon.txt.gz'
    path.write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))
    licence = next(Path(misaki.__file__).parent.parent.glob('misaki-*.dist-info/licenses/LICENSE'))
    (args.out / 'LICENSE-misaki.txt').write_text(NOTICE.format(version=version) + licence.read_text())
    print(f'{len(golds) + len(silvers)} of {n} entries kept; {len(text.encode())} bytes, {path.stat().st_size} gzipped')


if __name__ == '__main__':
    main()
