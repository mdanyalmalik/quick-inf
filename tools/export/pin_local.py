"""Prints the catalogue facts for an export in exports/: each file's size and sha256 (what Hugging Face lists for
it once uploaded), and for each graph its compute operations and largest weight tensor.

    tools/export/.venv/bin/python tools/export/pin_local.py exports/Qwen2.5-7B-Instruct-ONNX-WebGPU
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx

# Operations that move or reshape data rather than compute (not counted, as in the catalogue's computeOps).
NOT_COMPUTE = {
    'Constant', 'Shape', 'Reshape', 'Unsqueeze', 'Squeeze', 'Concat', 'Gather', 'Cast', 'Sub', 'ReduceSum', 'Range',
    'Expand', 'Equal', 'Where', 'ConstantOfShape', 'Identity', 'Transpose', 'Greater', 'If', 'Slice', 'Split', 'Less',
}
BYTES = {1: 4, 10: 2, 16: 2, 7: 8, 11: 8, 6: 4, 2: 1, 3: 1, 21: 0.5, 22: 0.5, 9: 1}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main(folder: Path):
    out = {'files': {}, 'graphs': {}}
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.name == 'README.md':
            continue
        rel = str(p.relative_to(folder))
        out['files'][rel] = [p.stat().st_size, sha256(p)]
        if p.suffix == '.onnx':
            m = onnx.load(str(p), load_external_data=False)
            ops = sum(1 for n in m.graph.node if n.op_type not in NOT_COMPUTE)
            largest = 0
            for t in m.graph.initializer:
                count = int(np.prod(t.dims)) if t.dims else 1
                largest = max(largest, int(BYTES.get(t.data_type, 4) * count))
            out['graphs'][rel] = {'ops': ops, 'largest': largest}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
