"""Pins WebLLM (MLC) builds for the catalogue. Writes src/catalogue/webllm-builds.ts (each build's
pins, sizes and tokenizer, for everything) and src/catalogue/webllm-files.ts (every file of each
build with its size and MD5, which only the worker needs).

For each build: the Hugging Face weight repo at its current commit, every file
WebLLM downloads (config, tokenizer, tensor-cache.json, weight shards) with
sizes and the MD5s listed in tensor-cache.json, and the compiled model library
pinned to a binary-mlc-llm-libs commit with its size and SRI hash.

Run from the repo root:  python3 tools/webllm/make_catalogue.py [--refresh]
Builds already in webllm-builds.ts keep their pinned commits (re-pinning changes
the files, so saved downloads and speed measurements would no longer apply);
new builds are pinned to the weight repo's current commit and the same library
commit. --refresh re-pins everything to the latest commits.
Re-run with --refresh (and re-test) when upgrading @mlc-ai/web-llm: the library
version (MODEL_VERSION) must match what that npm version expects.
"""
import base64, hashlib, json, os, re, sys, urllib.request

OUT = "src/catalogue/webllm-builds.ts"
OUT_FILES = "src/catalogue/webllm-files.ts"

# @mlc-ai/web-llm 0.2.85 expects model libraries built for "v0_2_84/base".
MODEL_VERSION = "v0_2_84/base"
LIBS_REPO = "mlc-ai/binary-mlc-llm-libs"

# WebLLM model id -> (library file name, vram_required_MB from WebLLM's prebuilt list)
# Not Gemma 3 1B: WebLLM 0.2.85 can't run its mixed local/global attention. Its cache is either
# a 512-token sliding window for every layer (crashes on longer prompts) or full attention for every
# layer (garbled replies past 512 tokens). Tested 2026-10-10.
BUILDS = {
    "SmolLM2-360M-Instruct-q4f16_1-MLC": ("SmolLM2-360M-Instruct-q4f16_1_cs1k-webgpu.wasm", 376.06),
    "SmolLM2-360M-Instruct-q4f32_1-MLC": ("SmolLM2-360M-Instruct-q4f32_1_cs1k-webgpu.wasm", 579.61),
    "Qwen2.5-0.5B-Instruct-q4f16_1-MLC": ("Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm", 944.62),
    "Qwen2.5-0.5B-Instruct-q4f32_1-MLC": ("Qwen2-0.5B-Instruct-q4f32_1_cs1k-webgpu.wasm", 1060.2),
    "Qwen3-0.6B-q4f16_1-MLC": ("Qwen3-0.6B-q4f16_1_cs1k-webgpu.wasm", 1403.34),
    "Qwen3-0.6B-q4f32_1-MLC": ("Qwen3-0.6B-q4f32_1_cs1k-webgpu.wasm", 1924.98),
    "SmolLM2-1.7B-Instruct-q4f16_1-MLC": ("SmolLM2-1.7B-Instruct-q4f16_1_cs1k-webgpu.wasm", 1774.19),
    "SmolLM2-1.7B-Instruct-q4f32_1-MLC": ("SmolLM2-1.7B-Instruct-q4f32_1_cs1k-webgpu.wasm", 2692.38),
    "Llama-3.2-1B-Instruct-q4f16_1-MLC": ("Llama-3.2-1B-Instruct-q4f16_1_cs1k-webgpu.wasm", 879.04),
    "Llama-3.2-1B-Instruct-q4f32_1-MLC": ("Llama-3.2-1B-Instruct-q4f32_1_cs1k-webgpu.wasm", 1128.82),
    "Qwen3-1.7B-q4f16_1-MLC": ("Qwen3-1.7B-q4f16_1_cs1k-webgpu.wasm", 2036.66),
    "Qwen3-1.7B-q4f32_1-MLC": ("Qwen3-1.7B-q4f32_1_cs1k-webgpu.wasm", 2635.44),
    # Added 2026-10-10 (WebLLM-only models).
    "Qwen2.5-1.5B-Instruct-q4f16_1-MLC": ("Qwen2-1.5B-Instruct-q4f16_1_cs1k-webgpu.wasm", 1629.75),
    "Qwen2.5-1.5B-Instruct-q4f32_1-MLC": ("Qwen2-1.5B-Instruct-q4f32_1_cs1k-webgpu.wasm", 1888.97),
    "Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC": ("Qwen2-1.5B-Instruct-q4f16_1_cs1k-webgpu.wasm", 1629.75),
    "Qwen2.5-Coder-1.5B-Instruct-q4f32_1-MLC": ("Qwen2-1.5B-Instruct-q4f32_1_cs1k-webgpu.wasm", 1888.97),
    "Qwen3.5-0.8B-q4f16_1-MLC": ("Qwen3.5-0.8B-q4f16_1_cs1k-webgpu.wasm", 1629.49),
    "Qwen3.5-0.8B-q4f32_1-MLC": ("Qwen3.5-0.8B-q4f32_1_cs1k-webgpu.wasm", 1894.19),
    "Qwen3.5-2B-q4f16_1-MLC": ("Qwen3.5-2B-q4f16_1_cs1k-webgpu.wasm", 2245.44),
    "Qwen3.5-2B-q4f32_1-MLC": ("Qwen3.5-2B-q4f32_1_cs1k-webgpu.wasm", 2591.55),
    "gemma-2-2b-it-q4f16_1-MLC": ("gemma-2-2b-it-q4f16_1_cs1k-webgpu.wasm", 1895.3),
    "gemma-2-2b-it-q4f32_1-MLC": ("gemma-2-2b-it-q4f32_1_cs1k-webgpu.wasm", 2508.75),
    "Llama-3.2-3B-Instruct-q4f16_1-MLC": ("Llama-3.2-3B-Instruct-q4f16_1_cs1k-webgpu.wasm", 2263.69),
    "Llama-3.2-3B-Instruct-q4f32_1-MLC": ("Llama-3.2-3B-Instruct-q4f32_1_cs1k-webgpu.wasm", 2951.51),
    "Qwen2.5-3B-Instruct-q4f16_1-MLC": ("Qwen2.5-3B-Instruct-q4f16_1_cs1k-webgpu.wasm", 2504.76),
    "Qwen2.5-3B-Instruct-q4f32_1-MLC": ("Qwen2.5-3B-Instruct-q4f32_1_cs1k-webgpu.wasm", 2893.64),
    "Phi-3.5-mini-instruct-q4f16_1-MLC": ("Phi-3.5-mini-instruct-q4f16_1_cs1k-webgpu.wasm", 3672.07),
    "Phi-3.5-mini-instruct-q4f32_1-MLC": ("Phi-3.5-mini-instruct-q4f32_1_cs1k-webgpu.wasm", 5483.12),
    "Phi-4-mini-instruct-q4f16_1-MLC": ("Phi-4-mini-instruct-q4f16_1_cs1k-webgpu.wasm", 3437.58),
    "Phi-4-mini-instruct-q4f32_1-MLC": ("Phi-4-mini-instruct-q4f32_1_cs1k-webgpu.wasm", 4220.59),
    "Qwen3-4B-q4f16_1-MLC": ("Qwen3-4B-q4f16_1_cs1k-webgpu.wasm", 3431.59),
    "Qwen3-4B-q4f32_1-MLC": ("Qwen3-4B-q4f32_1_cs1k-webgpu.wasm", 4327.71),
    # Large tier, added 2026-10-10: 4-9B, 3.9-8.4 GB of GPU memory (desktops).
    "Qwen3.5-4B-q4f16_1-MLC": ("Qwen3.5-4B-q4f16_1_cs1k-webgpu.wasm", 3867.82),
    "Qwen3.5-4B-q4f32_1-MLC": ("Qwen3.5-4B-q4f32_1_cs1k-webgpu.wasm", 4680.36),
    "Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC": ("Qwen2-7B-Instruct-q4f16_1_cs1k-webgpu.wasm", 5106.67),
    "Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC": ("Qwen2-7B-Instruct-q4f32_1_cs1k-webgpu.wasm", 5900.09),
    "Llama-3.1-8B-Instruct-q4f16_1-MLC": ("Llama-3_1-8B-Instruct-q4f16_1_cs1k-webgpu.wasm", 5001),
    "Llama-3.1-8B-Instruct-q4f32_1-MLC": ("Llama-3_1-8B-Instruct-q4f32_1_cs1k-webgpu.wasm", 6101.01),
    "Qwen3-8B-q4f16_1-MLC": ("Qwen3-8B-q4f16_1_cs1k-webgpu.wasm", 5695.78),
    "Qwen3-8B-q4f32_1-MLC": ("Qwen3-8B-q4f32_1_cs1k-webgpu.wasm", 6852.55),
    "gemma-2-9b-it-q4f16_1-MLC": ("gemma-2-9b-it-q4f16_1_cs1k-webgpu.wasm", 6422.01),
    "gemma-2-9b-it-q4f32_1-MLC": ("gemma-2-9b-it-q4f32_1_cs1k-webgpu.wasm", 8383.33),
    "Qwen3.5-9B-q4f16_1-MLC": ("Qwen3.5-9B-q4f16_1_cs1k-webgpu.wasm", 6433.01),
    "Qwen3.5-9B-q4f32_1-MLC": ("Qwen3.5-9B-q4f32_1_cs1k-webgpu.wasm", 7544.74),
    # Experimental (untested: no machine here has the ~31 GB of GPU memory it needs). 3-bit, fp16 only.
    "Llama-3.1-70B-Instruct-q3f16_1-MLC": ("Llama-3_1-70B-Instruct-q3f16_1_cs1k-webgpu.wasm", 31153.13),
}


def get(url, accept_json=True):
    req = urllib.request.Request(url, headers={"User-Agent": "quick-inf-catalogue"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    return json.loads(data) if accept_json else data


def sri(data):
    return "sha256-" + base64.b64encode(hashlib.sha256(data).digest()).decode()


def pinned():
    """Builds in the current webllm-builds.ts: model id -> (revision, library commit)."""
    if "--refresh" in sys.argv or not os.path.exists(OUT):
        return {}
    text = open(OUT).read()
    builds = json.loads(text[text.index("= {") + 2 : text.rindex(" satisfies")])
    lib_commit = lambda b: re.search(rf"{LIBS_REPO}/([0-9a-f]{{40}})/", b["modelLib"]["url"]).group(1)
    return {k: (b["revision"], lib_commit(b)) for k, b in builds.items()}


def main():
    pins = pinned()
    commits = {c for _, c in pins.values()}
    assert len(commits) <= 1, f"builds use several library commits: {commits}"
    libs_commit = commits.pop() if commits else get(f"https://api.github.com/repos/{LIBS_REPO}/commits?per_page=1")[0]["sha"]
    out = {}
    all_files = {}
    for model_id, (lib_name, vram_mb) in BUILDS.items():
        repo = f"mlc-ai/{model_id}"
        revision = pins[model_id][0] if model_id in pins else get(f"https://huggingface.co/api/models/{repo}/revision/main")["sha"]
        base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        config_bytes = get(base + "mlc-chat-config.json", accept_json=False)
        config = json.loads(config_bytes)
        tree = {f["path"]: f["size"] for f in get(f"https://huggingface.co/api/models/{repo}/tree/{revision}") if f["type"] == "file"}
        cache = get(base + "tensor-cache.json")
        tokenizer_file = "tokenizer.json" if "tokenizer.json" in config["tokenizer_files"] else config["tokenizer_files"][0]
        tokenizer_bytes = get(base + tokenizer_file, accept_json=False)
        files = [
            {"path": "mlc-chat-config.json", "sizeBytes": len(config_bytes)},
            {"path": tokenizer_file, "sizeBytes": len(tokenizer_bytes), "sha256": hashlib.sha256(tokenizer_bytes).hexdigest()},
            {"path": "tensor-cache.json", "sizeBytes": tree["tensor-cache.json"]},
        ]
        for rec in cache["records"]:
            assert tree[rec["dataPath"]] == rec["nbytes"], (model_id, rec["dataPath"])
            files.append({"path": rec["dataPath"], "sizeBytes": rec["nbytes"], "md5": rec["md5sum"]})
        lib_url = f"https://raw.githubusercontent.com/{LIBS_REPO}/{libs_commit}/web-llm-models/{MODEL_VERSION}/{lib_name}"
        lib = get(lib_url, accept_json=False)
        out[model_id] = {
            "repo": repo,
            "revision": revision,
            "modelLib": {"url": lib_url, "sizeBytes": len(lib), "integrity": sri(lib)},
            "integrity": {"config": sri(config_bytes), "tokenizer": {tokenizer_file: sri(tokenizer_bytes)}},
            "tokenizer": files[1],
            "downloadBytes": sum(f["sizeBytes"] for f in files) + len(lib),
            "weightBytes": sum(f["sizeBytes"] for f in files if "md5" in f),
            "vramMB": vram_mb,
            "contextWindow": config["context_window_size"],
            "prefillChunk": config.get("prefill_chunk_size"),
            "vocabSize": config["vocab_size"],
            "convTemplate": config["conv_template"].get("name") if isinstance(config["conv_template"], dict) else config["conv_template"],
        }
        all_files[model_id] = files
        total = sum(f["sizeBytes"] for f in files) + len(lib)
        print(f"{model_id:42} {'pinned' if model_id in pins else 'new   '} {revision[:10]} {len(files):3} files {total / 1e6:8.1f} MB  ctx {config['context_window_size']}", file=sys.stderr)

    with open(OUT, "w") as f:
        f.write("// Generated by tools/webllm/make_catalogue.py; do not edit by hand.\n")
        f.write("// Pinned WebLLM (MLC) builds: weight repo commit, download sizes, tokenizer, and the\n")
        f.write(f"// compiled model library ({MODEL_VERSION}, {LIBS_REPO}@{libs_commit[:10]}).\n\n")
        f.write("import type { WebLlmBuild } from './models';\n\n")
        f.write("export const WEBLLM_BUILDS = ")
        f.write(json.dumps(out, indent=2))
        f.write(" satisfies Record<string, WebLlmBuild>;\n")

    with open(OUT_FILES, "w") as f:
        f.write("// Generated by tools/webllm/make_catalogue.py; do not edit by hand.\n")
        f.write("// Every file WebLLM downloads for each pinned build (webllm-builds.ts), with the MD5s its\n")
        f.write("// tensor-cache.json lists for weight shards. Worker only: for the storage panel and checks.\n\n")
        f.write("import type { WebLlmFile } from './models';\n\n")
        f.write("export const WEBLLM_FILES: Record<string, WebLlmFile[]> = {\n")
        for model_id, files in all_files.items():
            f.write(f"  {json.dumps(model_id)}: [\n")
            for file in files:
                f.write("    " + json.dumps(file, separators=(", ", ": ")) + ",\n")
            f.write("  ],\n")
        f.write("};\n")


main()
