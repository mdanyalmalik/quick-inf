// Runs the browser G2P (src/runtime/speech/g2p) under Node, for the export and evaluation scripts. Reads a JSON list
// of texts on stdin and prints JSON: per text, what each TTS frontend would feed its model.
//   node tools/export/g2p_cli.mjs [frontends|misaki|ipa] < texts.json
// frontends: { kitten, inflect, paradee } chunk lists; misaki: Misaki strings without fallback (unknown words '❓');
// ipa: per text, the eSpeak-style IPA words.
import { readFileSync } from 'node:fs';
import { registerHooks } from 'node:module';

// Node strips TypeScript types itself; the app's imports leave out the .ts extension, so add it.
registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith('.') && !/\.[cm]?[jt]s$/.test(specifier) && context.parentURL?.endsWith('.ts')) {
      return next(`${specifier}.ts`, context);
    }
    return next(specifier, context);
  },
});

const root = new URL('../../', import.meta.url);
const g2p = await import(new URL('src/runtime/speech/g2p/index.ts', root).href);
const lexicon = await g2p.loadLexicon(readFileSync(new URL('exports/speech/g2p/en-us-lexicon.txt.gz', root)));
const g = new g2p.EnglishG2P(lexicon);
const texts = JSON.parse(readFileSync(0, 'utf8'));
const mode = process.argv[2] ?? 'frontends';
const paradeeConfig = new URL('tools/export/.cache/tts/paradee/config.json', root);
const vocab = mode === 'frontends' ? JSON.parse(readFileSync(paradeeConfig, 'utf8')).vocab : {};
const run = {
  frontends: (t) => ({ kitten: g.kitten(t), inflect: g.inflect(t), paradee: g.paradee(t, vocab) }),
  misaki: (t) => g.misaki(t, { fallback: false, unk: '❓' }),
  ipa: (t) => g.ipa(t),
}[mode];
process.stdout.write(JSON.stringify(texts.map(run)));
