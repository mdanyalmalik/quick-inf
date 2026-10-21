"""Generates the TTS evaluation sentences with the browser G2P (src/runtime/speech/g2p, run under Node by
g2p_cli.mjs) in place of each model's own eSpeak / Misaki step, through the model's real Python pipeline, and writes
tools/export/.cache/tts-eval/out/<model>/<variant>/{0..7}.wav and stats.json like tts_gen.py. The int4 / int8 builds
the q4 / q8 rows use: Kitten and Inflect q4, Paradee q8. Score with
    tools/export/.cache/mlx-venv/bin/python tools/export/.cache/tts-eval/tts_score.py <model>

    tools/export/.cache/tts-venv/bin/python tools/export/g2p_eval.py MODEL [VARIANT]
MODEL: kitten-nano kitten-micro kitten-mini inflect-micro inflect-nano paradee; VARIANT defaults to g2p.
"""

import json
import resource
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'tools/export/.cache/tts-eval'
sys.path.insert(0, str(EVAL))
import tts_gen  # noqa: E402


def ts_frontends(texts):
    out = subprocess.run(['node', str(ROOT / 'tools/export/g2p_cli.mjs'), 'frontends'], input=json.dumps(texts),
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def chunk_table(results, key):
    return {c['text']: c for r in results for c in r[key]}


def patch_kitten(table):
    """phonemizer's eSpeak backend → the TS chunk's phonemes (Kitten re-tokenizes them; the ids must agree)."""
    from kittentts import onnx_model

    class Backend:
        def __init__(self, *a, **k):
            pass

        def phonemize(self, texts):
            # Keyed without spaces: the TS port keeps the space the package's time pattern swallows.
            return [by_letters[t.replace(' ', '')]['phonemes'] for t in texts]

    by_letters = {k.replace(' ', ''): c for k, c in table.items()}
    onnx_model.phonemizer.backend.EspeakBackend = Backend
    cleaner = onnx_model.TextCleaner()
    for c in table.values():
        ids = cleaner(' '.join(onnx_model.basic_english_tokenize(c['phonemes'])))
        assert [0, *ids, 0] == c['ids'], c


def patch_inflect(size, table):
    sys.path.insert(0, str(tts_gen.C / f'inflect-{size}/onnx'))
    import inference_onnx
    from inflect_nano_v2_frontend import normalize_text

    def frontend(text):
        c = table[normalize_text(text)]
        assert inference_onnx.phonemes_to_tokens(c['phonemes'])[0].tolist() == c['ids'], c
        return types.SimpleNamespace(phoneme_text=c['phonemes'])

    inference_onnx.run_vits_frontend = frontend


def patch_paradee(table):
    from paradee import tts
    tts.Paradee.phonemize = lambda self, sentence: table[sentence]['phonemes']


def main():
    model = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else 'g2p'
    out = EVAL / 'out' / model / variant
    results = ts_frontends(['Warm up.', *tts_gen.SENTENCES])
    if model.startswith('kitten-'):
        patch_kitten(chunk_table(results, 'kitten'))
        say = tts_gen.kitten(model.removeprefix('kitten-'), 'q4')
    elif model.startswith('inflect-'):
        size = model.removeprefix('inflect-')
        patch_inflect(size, chunk_table(results, 'inflect'))
        say = tts_gen.inflect(size, True)
    else:
        patch_paradee(chunk_table(results, 'paradee'))
        say = tts_gen.paradee('q8')
    out.mkdir(parents=True, exist_ok=True)
    say('Warm up.')
    stats = []
    for i, text in enumerate(tts_gen.SENTENCES):
        start = time.perf_counter()
        wav, rate = say(text)
        elapsed = time.perf_counter() - start
        sf.write(out / f'{i}.wav', wav.astype(np.float32), rate)
        stats.append({'gen_s': elapsed, 'audio_s': len(wav) / rate, 'rate': rate})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    (out / 'stats.json').write_text(json.dumps({'runs': stats, 'peak_rss_mb': rss}))
    print(model, variant, f'{sum(s["audio_s"] for s in stats):.1f} s of audio')


if __name__ == '__main__':
    main()
