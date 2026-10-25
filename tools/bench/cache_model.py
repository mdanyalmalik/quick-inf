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
FILE = re.compile(
    r"\{\s*path: '([^']+)',\s*sizeBytes: (\d+),\s*sha256: '([0-9a-f]{64})'"
    r"(?:,\s*from: \{\s*repo: '([^']+)',\s*revision: '([0-9a-f]+)',\s*path: '([^']+)',?\s*\})?,?\s*\}"
)


def model_block(model_id: str) -> str:
    text = (ROOT / 'src/catalogue/models.ts').read_text()
    at = text.find(f"id: '{model_id}',")
    if at < 0:
        sys.exit(f'No model {model_id} in models.ts')
    start = text.rfind('\nexport const ', 0, at)
    end = text.find('\n};\n', at)
    return text[start:end]


def const_block(name: str) -> str:
    """The entry `export const NAME`, for an entry spread from it (`...NAME,`)."""
    text = (ROOT / 'src/catalogue/models.ts').read_text()
    start = text.find(f'\nexport const {name}:')
    if start < 0:
        sys.exit(f'No {name} in models.ts')
    return text[start:text.find('\n};\n', start)]


def first_artifact(block: str, artifact_id: str | None) -> str:
    artifacts = block[block.find('artifacts:'):]
    if artifact_id:
        artifacts = artifacts[artifacts.find(f"id: '{artifact_id}'"):]
    return artifacts[: artifacts.find("\n    },") if "\n    }," in artifacts else len(artifacts)]


def main(model_id: str, artifact_id: str | None):
    block = model_block(model_id)
    # Comment lines (explaining a source choice) don't count.
    source = re.search(r"source: \{\s*host: 'huggingface',\s*repo: ([^,]+),\s*revision: ([^\s,]+),?\s*\}", re.sub(r"\n\s*//[^\n]*", "", block))
    if not source:
        sys.exit('No literal source in the entry (spread from another model?)')
    repo = source.group(1).strip("'").replace('${OWN_EXPORTS}', 'danyalmalik').strip('`')
    revision = source.group(2).strip("'")
    # An entry spread from another (`...NAME,`) takes its tokenizer, and an artifact spread from another's
    # (`...NAME.artifacts[0]!,`) its files, except those it names itself (by path).
    base = re.search(r"\n  \.\.\.(\w+),", block)
    tokenizer = FILE.search(block[block.find('tokenizer:'):]) if 'tokenizer:' in block else FILE.search((b := const_block(base.group(1)))[b.find('tokenizer:'):])
    first = first_artifact(block, artifact_id)
    files = list(FILE.finditer(first))
    spread = re.search(r"\.\.\.(\w+)\.artifacts\[(\d+)\]", first)
    if spread:
        own = {f.group(1) for f in files}
        inherited = first_artifact(const_block(spread.group(1)), None)
        files += [f for f in FILE.finditer(inherited) if f.group(1) not in own]
    for f in [tokenizer] + files:
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
