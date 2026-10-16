"""Downloads a catalogue model's files (its tokenizer and first artifact, or the one named) into the dev server's local
model cache, tools/reference/.cache/<owner>__<repo>/<revision>/, checking each file's size and SHA-256 against
src/catalogue/models.ts. For the speed bench (tools/bench/bench.mjs).

    tools/export/.venv/bin/python tools/bench/cache_model.py <model id> [artifact id]
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILE = re.compile(r"\{ path: '([^']+)', sizeBytes: (\d+), sha256: '([0-9a-f]{64})'(?:, from: \{ repo: '([^']+)', revision: '([0-9a-f]+)', path: '([^']+)' \})? \}")


def model_block(model_id: str) -> str:
    text = (ROOT / 'src/catalogue/models.ts').read_text()
    at = text.find(f"id: '{model_id}',")
    if at < 0:
        sys.exit(f'No model {model_id} in models.ts')
    start = text.rfind('\nexport const ', 0, at)
    end = text.find('\n};\n', at)
    return text[start:end]


def main(model_id: str, artifact_id: str | None):
    block = model_block(model_id)
    source = re.search(r"source: \{ host: 'huggingface', repo: ([^,]+), revision: ([^ ]+) \}", block)
    if not source:
        sys.exit('No literal source in the entry (spread from another model?)')
    repo = source.group(1).strip("'").replace('${OWN_EXPORTS}', 'danyalmalik').strip('`')
    revision = source.group(2).strip("'")
    tokenizer = FILE.search(block[block.find('tokenizer:'):])
    artifacts = block[block.find('artifacts:'):]
    if artifact_id:
        artifacts = artifacts[artifacts.find(f"id: '{artifact_id}'"):]
    first = artifacts[: artifacts.find("\n    },") if "\n    }," in artifacts else len(artifacts)]
    files = [tokenizer] + list(FILE.finditer(first))
    for f in files:
        path, size, sha, from_repo, from_rev, from_path = f.groups()
        url = f'https://huggingface.co/{from_repo or repo}/resolve/{from_rev or revision}/{from_path or path}'
        dest = ROOT / 'tools/reference/.cache' / f"{repo.replace('/', '__')}" / revision / path
        if dest.exists() and dest.stat().st_size == int(size):
            print('have', path)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print('get ', path, f'{int(size) / 1e9:.2f} GB', flush=True)
        subprocess.run(['curl', '-sSL', '--fail', '-o', str(dest) + '.part', url], check=True)
        h = hashlib.sha256()
        with open(str(dest) + '.part', 'rb') as fh:
            while chunk := fh.read(1 << 24):
                h.update(chunk)
        if h.hexdigest() != sha:
            sys.exit(f'{path}: sha256 mismatch')
        Path(str(dest) + '.part').rename(dest)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
