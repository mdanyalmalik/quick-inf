"""Prints the catalogue entry (src/catalogue/models.ts) for a model's ONNX file, measured from
the pinned files in the reference cache: the graph and any external weight files, Qwen3.5's
separate embedding graph, the KV-cache type, the largest single weight tensor (checked against a
GPU's buffer limit) and the compute-op count (the per-token fixed cost in speed estimates).

Fetch the files first (fetch_model.py; --with-q4 for the fp32-maths files), then:
  tools/reference/.venv/bin/python -I tools/reference/onnx_artifact.py <model> [--q4]
"""

import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_model import MODELS, cache_dir  # noqa: E402

# Shape bookkeeping and control flow: not counted as compute (matches the catalogue's earlier counts).
NOT_COMPUTE = {
    "Constant", "Shape", "Reshape", "Unsqueeze", "Squeeze", "Concat", "Gather", "Cast", "Sub", "ReduceSum", "Range",
    "Expand", "Equal", "Where", "ConstantOfShape", "Identity", "Transpose", "Greater", "If", "Slice", "Split", "Less",
}
# Bytes per element by ONNX data type (INT4/UINT4 pack two per byte).
ELEMENT_BYTES = {1: 4, 10: 2, 16: 2, 7: 8, 11: 8, 6: 4, 2: 1, 3: 1, 21: 0.5, 22: 0.5}


def measure(path: Path):
    model = onnx.load(str(path), load_external_data=False)
    ops = sum(1 for n in model.graph.node if n.op_type not in NOT_COMPUTE)
    largest = max(int(ELEMENT_BYTES.get(t.data_type, 4) * int(np.prod(t.dims) if t.dims else 1)) for t in model.graph.initializer)
    kv = next((i for i in model.graph.input if i.name.startswith("past_key_values.")), None)
    kv_type = "float16" if kv is not None and kv.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 else "float32"
    return ops, largest, kv_type


def main():
    precision = "q4" if "--q4" in sys.argv else "q4f16"
    for model_id in [a for a in sys.argv[1:] if not a.startswith("--")]:
        files = {**MODELS[model_id]["files"], **MODELS[model_id]["optional"]}
        ref = lambda path: f"{{ path: '{path}', sizeBytes: {files[path][0]}, sha256: '{files[path][1]}' }}"
        refs = lambda paths: "[" + ", ".join(ref(p) for p in paths) + "]"
        graph = next(p for p in files if p.endswith(f"_{precision}.onnx") and "embed_tokens" not in p and "vision" not in p)
        ops, largest, kv_type = measure(cache_dir(model_id) / graph)
        lines = [
            "{",
            f"  id: 'onnx-{precision}',",
            "  engine: 'onnx-webgpu',",
            "  format: 'onnx',",
            f"  precision: '{precision}',",
            f"  requiredGpuFeatures: {['shader-f16'] if precision == 'q4f16' else []},".replace('"', "'"),
            f"  graph: {ref(graph)},",
        ]
        data = sorted(p for p in files if p.startswith(graph + "_data"))
        if data:
            lines.append(f"  externalData: {refs(data)},")
        embed = next((p for p in files if "embed_tokens" in p and p.endswith(f"_{precision}.onnx")), None)
        if embed:
            embed_ops, embed_largest, _ = measure(cache_dir(model_id) / embed)
            ops, largest = ops + embed_ops, max(largest, embed_largest)
            lines.append(f"  embeddings: {{ graph: {ref(embed)}, externalData: {refs(sorted(p for p in files if p.startswith(embed + '_data')))} }},")
        lines += [f"  kvCacheType: '{kv_type}',", f"  largestTensorBytes: {largest},", f"  computeOps: {ops},", "},"]
        print(f"// {model_id} ({MODELS[model_id]['repo']}@{MODELS[model_id]['revision'][:10]})")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
