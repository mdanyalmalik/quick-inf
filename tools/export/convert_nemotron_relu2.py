"""Converts a Nemotron 3 Nano export made with SwiGLU's duplicated up rows into one using QMoE's relu2 activation.

The first exports (export_nemotron.py before relu2) ran the experts' squared ReLU as SwiGLU with each up-projection
row twice ([u0, u0, u1, u1, ...]) and a large alpha, as ONNX Runtime's WebGPU QMoE only had SwiGLU. Quick
Inference's ONNX Runtime build (github.com/mdanyalmalik/onnxruntime-web-kernels) has activation_type="relu2" (the
name ONNX Runtime's CUDA MoE uses), so the copies can go: every second row of the fc1 weights, scales and bias is
dropped (the pairs were quantized identically, so this is exact), the QMoE nodes get relu2, and the expert weights
shrink by about a third. Tensors are copied one at a time from the original data file.

Usage: tools/export/.venv/bin/python -I tools/export/convert_nemotron_relu2.py <source dir> <output dir>
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

ALIGN = 64
DTYPES = {TensorProto.UINT8: np.uint8, TensorProto.FLOAT16: np.float16, TensorProto.FLOAT: np.float32}


def external(tensor):
    info = {e.key: e.value for e in tensor.external_data}
    return info['location'], int(info.get('offset', 0)), int(info['length'])


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(src / 'model.onnx', load_external_data=False)
    graph = model.graph

    # fc1 inputs of every QMoE node (qweight 2, scales 3, bias 4), whose rows come in identical pairs.
    halved = set()
    for node in graph.node:
        if node.op_type != 'QMoE':
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get('activation_type') != b'swiglu' or attrs.get('swiglu_fusion') != 1:
            raise SystemExit(f'{node.name}: not the duplicated-row SwiGLU layout')
        halved.update(n for n in node.input[2:5] if n)
        kept = [a for a in node.attribute if a.name not in ('activation_type', 'swiglu_fusion', 'activation_alpha', 'activation_beta')]
        del node.attribute[:]
        node.attribute.extend(kept + [helper.make_attribute('activation_type', 'relu2')])

    location = 'model.onnx.data'
    data = np.memmap(src / location, dtype=np.uint8, mode='r')
    written = 0
    with open(out / location, 'wb') as f:
        for tensor in graph.initializer:
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            loc, offset, length = external(tensor)
            if loc != location:
                raise SystemExit(f'{tensor.name}: unexpected data file {loc}')
            raw = data[offset : offset + length]
            if tensor.name in halved:
                dims = list(tensor.dims)
                array = np.frombuffer(raw, dtype=DTYPES[tensor.data_type]).reshape(dims)
                pairs = array[:, 0::2]
                if not np.array_equal(pairs, array[:, 1::2]):
                    raise SystemExit(f'{tensor.name}: rows are not identical pairs')
                raw = np.ascontiguousarray(pairs).view(np.uint8).reshape(-1)
                dims[1] //= 2
                del tensor.dims[:]
                tensor.dims.extend(dims)
            pad = (-written) % ALIGN
            f.write(b'\0' * pad)
            written += pad
            f.write(raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw))
            del tensor.external_data[:]
            for key, value in (('location', location), ('offset', str(written)), ('length', str(raw.nbytes))):
                tensor.external_data.add(key=key, value=value)
            written += raw.nbytes
    onnx.save(model, out / 'model.onnx', save_as_external_data=False)
    print(f'{len(halved)} fc1 tensors halved; data {written / 1e9:.2f} GB (was {data.nbytes / 1e9:.2f} GB)')


if __name__ == '__main__':
    main()
