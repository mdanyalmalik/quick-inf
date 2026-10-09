"""Confirms models rewritten by externalizeWeights (src/runtime/onnx/external-weights.ts)
produce exactly the original model's outputs.

Run the TypeScript test first (it writes <model>.onnx.external-graph next to each
cached model file), then:
  tools/reference/.venv/bin/python -I tools/reference/check_external.py
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_model import MODELS, cache_dir  # noqa: E402


def run(path: Path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in s.get_inputs()}
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    feeds = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
    if "position_ids" in inputs:
        feeds["position_ids"] = np.arange(len(ids), dtype=np.int64)[None, :]
    if "attention_bias" in inputs:  # added by the rewrite in place of the mask subgraph
        dtype = np.float16 if inputs["attention_bias"].type == "tensor(float16)" else np.float32
        n = len(ids)
        bias = np.triu(np.full((n, n), np.finfo(dtype).min, dtype=dtype), k=1)
        feeds["attention_bias"] = bias[None, None, :, :]
    for name, i in inputs.items():
        if name.startswith("past_key_values."):
            dtype = np.float16 if i.type == "tensor(float16)" else np.float32
            feeds[name] = np.zeros((1, i.shape[1], 0, i.shape[3]), dtype=dtype)
    return s.run(["logits"], feeds)[0]


def main():
    checked = 0
    for model_id in MODELS:
        for graph in sorted(cache_dir(model_id).glob("onnx/*.external-graph")):
            original = graph.with_name(graph.name.removesuffix(".external-graph"))
            # ONNX Runtime resolves external data relative to the model's directory and wants a .onnx name.
            renamed = graph.with_suffix(".external.onnx")
            renamed.write_bytes(graph.read_bytes())
            a, b = run(original), run(renamed)
            same = np.array_equal(a, b)
            print(f"{'IDENTICAL' if same else 'DIFFERENT'} {model_id} {original.name}: max diff {float(np.max(np.abs(a - b))):.6f}")
            checked += 1
            renamed.unlink()
    if checked == 0:
        raise SystemExit("no rewritten graphs found; run the vitest external-weights test first")


if __name__ == "__main__":
    main()
