"""A VibeVoice Realtime voice preset (the prompt caches tools/export/export_vibevoice.py saves as .npz) as one
little-endian float32 file and a JSON index, for the browser: `NAME.bin` holds every array back to back, `NAME.json`
maps each array's name to its offset (in values) and shape.

    python tools/export/vibevoice_voice.py VOICE.npz OUT_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    voice = np.load(src)
    index, parts, offset = {}, [], 0
    for name in voice.files:
        a = voice[name].astype('<f4')
        index[name] = {'offset': offset, 'shape': list(a.shape)}
        parts.append(a.ravel())
        offset += a.size
    out.mkdir(parents=True, exist_ok=True)
    np.concatenate(parts).tofile(out / f'{src.stem}.bin')
    (out / f'{src.stem}.json').write_text(json.dumps(index, separators=(',', ':')))
    print(f'{src.stem}: {len(index)} arrays, {offset * 4} bytes')


if __name__ == '__main__':
    main()
