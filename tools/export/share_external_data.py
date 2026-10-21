"""Moves the large initializers of several ONNX graphs into one external-data file they share: tensors with identical
contents (type, shape and bytes) are stored once, whatever they are called in each graph, and every graph points at
them by offset. For model families exported as several graphs over one set of weights (MOSS-TTS-Nano's prefill and
decode-step graphs, its codec's full and streaming decoders), where quantizing each graph on its own embeds a copy
of the shared weights in every file.

Writes OUT_DIR/<name of each input> and OUT_DIR/DATA_NAME; tensors under --min-bytes stay inside the graphs.

    tools/export/.venv/bin/python tools/export/share_external_data.py DATA_NAME OUT_DIR IN.onnx [IN.onnx ...] [--min-bytes 1024]
"""

import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import TensorProto

ALIGN = 4096


def share(data_name: str, out_dir: Path, sources: list[Path], min_bytes: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / data_name
    stored = {}  # (type, shape, sha256) -> (offset, length)
    referenced = 0
    with open(data_path, 'wb') as data:
        for src in sources:
            model = onnx.load(str(src))
            moved = kept = 0
            for t in model.graph.initializer:
                if t.data_location == TensorProto.EXTERNAL or len(t.raw_data) < min_bytes:
                    kept += 1
                    continue
                raw = t.raw_data
                key = (t.data_type, tuple(t.dims), hashlib.sha256(raw).hexdigest())
                if key not in stored:
                    offset = -(-data.tell() // ALIGN) * ALIGN
                    data.write(b'\0' * (offset - data.tell()))
                    data.write(raw)
                    stored[key] = (offset, len(raw))
                offset, length = stored[key]
                referenced += length
                t.ClearField('raw_data')
                t.data_location = TensorProto.EXTERNAL
                del t.external_data[:]
                for k, v in (('location', data_name), ('offset', str(offset)), ('length', str(length))):
                    entry = t.external_data.add()
                    entry.key, entry.value = k, v
                moved += 1
            dst = out_dir / src.name
            onnx.save(model, str(dst))
            print(f'{src.name}: {moved} tensors -> {data_name}, {kept} kept inline; graph {dst.stat().st_size / 1e6:.1f} MB')
    size = data_path.stat().st_size
    print(f'{data_name}: {len(stored)} tensors, {size / 1e6:.1f} MB (the graphs reference {referenced / 1e6:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name')
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--min-bytes', type=int, default=1024)
    args = parser.parse_args()
    share(args.data_name, args.out_dir, args.sources, args.min_bytes)
