"""Builds our own ONNX exports of chat models for the WebGPU engine, ready to upload to Hugging Face.

Each model is converted with ONNX Runtime GenAI's model builder (int4 weights, k-quant, block size 32,
float16 activations and KV cache) into exports/<repo name>/, with the original licence files, any notice
the licence asks for, and a model card naming the original and saying what was changed.

    tools/export/.venv/bin/python tools/export/export_llm.py qwen2.5-7b-instruct
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / '.cache'
EXPORTS = ROOT / 'exports'

QWEN_APACHE = 'Qwen2.5 is licensed under the Apache License 2.0; see LICENSE-Qwen2.5.'

MODELS = {
    'qwen2.5-7b-instruct': {
        'source': 'Qwen/Qwen2.5-7B-Instruct',
        'repo': 'Qwen2.5-7B-Instruct-ONNX-WebGPU',
        'license': 'apache-2.0',
        'licenseFiles': ['LICENSE'],
    },
    'qwen2.5-coder-7b-instruct': {
        'source': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'repo': 'Qwen2.5-Coder-7B-Instruct-ONNX-WebGPU',
        'license': 'apache-2.0',
        'licenseFiles': ['LICENSE'],
    },
    'deepseek-r1-distill-qwen-7b': {
        'source': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'repo': 'DeepSeek-R1-Distill-Qwen-7B-ONNX-WebGPU',
        'license': 'mit',
        'licenseFiles': ['LICENSE'],
        # Distilled from Qwen2.5-Math-7B, which is Apache 2.0.
        'extraLicense': ('Qwen/Qwen2.5-Math-7B', 'LICENSE', 'LICENSE-Qwen2.5'),
        'notes': [QWEN_APACHE],
    },
    'deepseek-r1-0528-qwen3-8b': {
        'source': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        'repo': 'DeepSeek-R1-0528-Qwen3-8B-ONNX-WebGPU',
        'license': 'mit',
        'licenseFiles': ['LICENSE'],
        # The config's YaRN leaves out beta_fast/beta_slow (the builder needs them; 32 and 1 are YaRN's defaults) and sets
        # attn_factor 0.878, which cancels YaRN's default magnitude scaling (1 + 0.1 ln 4) to exactly 1, as vLLM applies it.
        'ropePatch': {'beta_fast': 32, 'beta_slow': 1, 'mscale': 1.0},
        'changes': ['YaRN made explicit for the builder: beta_fast 32, beta_slow 1 (the defaults) and a rotary magnitude scale of 1.0 (the config\'s attn_factor × YaRN\'s default scaling).'],
    },
    'gemma-3-12b-it': {
        # Google's repo is gated; unsloth's copy has the same weights without the gate.
        'source': 'unsloth/gemma-3-12b-it',
        'original': 'google/gemma-3-12b-it',
        'repo': 'gemma-3-12b-it-ONNX-WebGPU',
        'license': 'gemma',
        'licenseFiles': [],
        'ourLicences': ['GEMMA_TERMS_OF_USE.md', 'GEMMA_PROHIBITED_USE_POLICY.md'],
        'writeFiles': {
            'NOTICE': 'Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\n',
        },
        'restrictions': (
            '\n## Use restrictions\n\n'
            'Gemma is provided under and subject to the Gemma Terms of Use (GEMMA_TERMS_OF_USE.md). By using these files you '
            'agree to those terms, including the use restrictions in their section 3.2: you may not use these files or '
            'anything made from them for the restricted uses in the Gemma Prohibited Use Policy '
            '(GEMMA_PROHIBITED_USE_POLICY.md). If you distribute them or anything made from them, you must pass these '
            'restrictions on in the same way.\n'
        ),
    },
}


def run(args):
    print('+', ' '.join(str(a) for a in args), flush=True)
    subprocess.run(args, check=True)


def download(repo: str, target: Path, include=None):
    args = [HERE / '.venv/bin/hf', 'download', repo, '--local-dir', target]
    if include:
        args += ['--include', *include]
    run(args)


def model_card(key: str, spec: dict, out: Path) -> str:
    source = spec.get('original', spec['source'])
    files = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != 'README.md')
    notes = '\n'.join(f'\n{n}' for n in spec.get('notes', []))
    restrictions = spec.get('restrictions', '')
    changes = ''.join(f'- {c}\n' for c in spec.get('changes', []))
    return f"""---
license: {spec['license']}
base_model:
- {source}
tags:
- onnx
- webgpu
- onnxruntime-genai
- quick-inference
---

# {spec['repo']}

[{source}](https://huggingface.co/{source}) converted to ONNX for running in a web browser with
ONNX Runtime Web's WebGPU backend, as used by Quick Inference.

## Changes from the original

- Converted to ONNX with ONNX Runtime GenAI's model builder (`-p int4 -e webgpu`).
- The weights of the linear layers are quantised to 4 bits (k-quant, block size 32; the LM head round-to-nearest, block size 32); the rest are float16.
- Activations and the KV cache are float16.
{changes}
These are modified files: their outputs are close to, but not identical to, the original model's.
{restrictions}
## Licence

Licensed under the same terms as the original model: see {', '.join(f for f in files if f.startswith(('LICENSE', 'NOTICE', 'GEMMA')))}.{notes}

## Files

{chr(10).join(f'- `{f}`' for f in files)}
"""


def export(key: str):
    spec = MODELS[key]
    src = CACHE / 'hf' / spec['source'].split('/')[1]
    out = EXPORTS / spec['repo']
    if not (src / 'config.json').exists():
        download(spec['source'], src)
    if 'ropePatch' in spec:
        config = json.loads((src / 'config.json').read_text())
        config['rope_scaling'].update(spec['ropePatch'])
        (src / 'config.json').write_text(json.dumps(config, indent=2))
    work = CACHE / 'build' / key
    # Run under the swap guard. The LM head stays out of k-quant (it needs tens of GB for a 150k-row matrix) and is
    # quantised afterwards by quantize_lm_head.py, a few thousand rows at a time.
    run([
        HERE / 'guarded.sh', HERE / '.venv/bin/python', '-m', 'onnxruntime_genai.models.builder',
        '-i', src, '-o', out, '-p', 'int4', '-e', 'webgpu', '-c', work,
        '--extra_options', 'algo_config=k_quant', 'block_size=32', 'nodes_to_exclude=/lm_head/MatMul', *spec.get('builderOptions', []),
    ])
    run([HERE / 'guarded.sh', HERE / '.venv/bin/python', HERE / 'quantize_lm_head.py', out])
    shutil.rmtree(work, ignore_errors=True)
    for name in spec['licenseFiles']:
        shutil.copy(src / name, out / name)
    for name in spec.get('ourLicences', []):
        shutil.copy(HERE / 'licences' / name, out / name)
    if 'extraLicense' in spec:
        repo, name, as_name = spec['extraLicense']
        tmp = CACHE / 'hf' / repo.split('/')[1]
        download(repo, tmp, [name])
        shutil.copy(tmp / name, out / as_name)
    for name, text in spec.get('writeFiles', {}).items():
        (out / name).write_text(text)
    (out / 'README.md').write_text(model_card(key, spec, out))
    sizes = {p.name: p.stat().st_size for p in sorted(out.iterdir())}
    print(json.dumps(sizes, indent=1))


if __name__ == '__main__':
    for key in sys.argv[1:]:
        export(key)
