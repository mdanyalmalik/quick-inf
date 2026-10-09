"""Download the pinned model files used by the reference scripts and tests.

Files go to tools/reference/.cache/<repo>/<revision>/ (gitignored). Each file is
checked against the size and SHA-256 recorded in src/catalogue/models.ts, so the
reference and the app always use byte-identical artifacts.

Usage: tools/reference/.venv/bin/python -I tools/reference/fetch_model.py [model ...] [--with-q4]
       (no model names = all models)
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

# Keep in sync with src/catalogue/models.ts.
MODELS = {
    "smollm2-360m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "revision": "a10cc1512eabd3dde888204e902eca88bddb4951",
        "files": {
            "tokenizer.json": (2104556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
            "tokenizer_config.json": (3764, "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"),
            "onnx/model_q4f16.onnx": (272737275, "cc63370efc2aca6d5307518b85162777132cc5b8d68eeb8154ea9b5fce09ad46"),
        },
        "optional": {
            "onnx/model_q4.onnx": (387943246, "a4ffda96e65beafc6f6cef0cbcf9fdc1cbdd79c230906bf3897d190547c7a596"),
        },
    },
    "qwen2.5-0.5b-instruct": {
        # The -ONNX-MHA export: fused attention (MultiHeadAttention, RotaryEmbedding).
        # The older onnx-community/Qwen2.5-0.5B-Instruct export does attention with
        # plain MatMuls, which ONNX Runtime's WebGPU kernel gets wrong on Adreno GPUs
        # for some shapes, and its q4f16 file overflows fp16.
        "repo": "onnx-community/Qwen2.5-0.5B-Instruct-ONNX-MHA",
        "revision": "b1e71b59c0c33ee49eb562303578961795624293",
        "files": {
            "tokenizer.json": (9116490, "bb5953180a154511cf4d0fb1fc68a3433f1458aa6db3851ac2609cf41e7d9b5b"),
            "tokenizer_config.json": (7849, "6a14243f0ae04884e597b3f032c05e15de9f417318fdcac24ab4e2680e691430"),
            "onnx/model_q4f16.onnx": (478355493, "2a8e924345337b83796bd98bc130ab0ec0163188d50be55a1bf14b95808c55b8"),
        },
        "optional": {
            "onnx/model_q4.onnx": (777326354, "60e105dee084373137dd810d0d5f47c3a0cfe14c2dd3c25f723fba3169e011f6"),
        },
    },
    "qwen3-0.6b": {
        "repo": "onnx-community/Qwen3-0.6B-ONNX",
        "revision": "1e0a4a196ecabdf9a879664110574563d3f372d3",
        "files": {
            "tokenizer.json": (9117040, "e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5"),
            "tokenizer_config.json": (10360, "982fa8771b0e9e5ebf7e943631c6fc4af301aa81b509e1ca517a7389e4a51d42"),
            "onnx/model_q4f16.onnx": (569789750, "9e33a5911974174761d0dfdcc0bec975d9c45af0eae5e9eb647b8ba9442a8f91"),
        },
        "optional": {
            "onnx/model_q4.onnx": (919096585, "d43d836fc5e240df9013733ccd214972c5d21bd9ec47e574e4f1e359cf90aed0"),
        },
    },
    "smollm2-1.7b-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "revision": "31b70e2e869a7173562077fd711b654946d38674",
        "files": {
            "tokenizer.json": (2104556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
            "tokenizer_config.json": (3764, "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"),
            "onnx/model_q4f16.onnx": (1108730338, "d94946187fb5f27579f3db4ba21fb7f7466c7cbd18956bd420d3981f75282f9c"),
        },
        "optional": {
            "onnx/model_q4.onnx": (1411969607, "467b7b8f62d99f184d3628d24b8d65c151e331695f6e9ea997616c4e279e9a51"),
        },
    },
}

CACHE_ROOT = Path(__file__).resolve().parent / ".cache"


def cache_dir(model_id: str) -> Path:
    m = MODELS[model_id]
    return CACHE_ROOT / m["repo"].replace("/", "__") / m["revision"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(model_id: str, rel: str, size: int, digest: str) -> Path:
    m = MODELS[model_id]
    dest = cache_dir(model_id) / rel
    if dest.exists() and dest.stat().st_size == size and sha256_of(dest) == digest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{rel}"
    print(f"downloading {model_id}/{rel} ({size / 1e6:.1f} MB)", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
    actual_size, actual = tmp.stat().st_size, sha256_of(tmp)
    if actual_size != size or actual != digest:
        raise SystemExit(f"{rel}: expected {size} bytes sha256 {digest}, got {actual_size} bytes sha256 {actual}")
    tmp.rename(dest)
    return dest


def main() -> None:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(MODELS)
    for model_id in names:
        files = dict(MODELS[model_id]["files"])
        if "--with-q4" in sys.argv:
            files.update(MODELS[model_id]["optional"])
        for rel, (size, digest) in files.items():
            print(f"ok {fetch(model_id, rel, size, digest)}", flush=True)


if __name__ == "__main__":
    main()
