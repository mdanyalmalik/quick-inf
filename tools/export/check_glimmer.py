"""Checks export_glimmer.py's graph against transformers' own Muse Glimmer text layers (MuseGlimmerTextDecoderLayer, eager
attention, float32) on the first N layers, run one layer at a time with Red Hat's 4-bit weights dequantized and the
embeddings and output layer rounded to 4 bits as the export rounds them, so only the graph's structure can differ:
logits of a prompt longer than the 2,048-token sliding window, then of a few tokens decoded against the cache.

    tools/export/.venv/bin/python tools/export/export_glimmer.py --layers 4 --out DIR
    tools/export/.venv/bin/python tools/export/check_glimmer.py DIR --layers 4
"""

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer, MuseGlimmerTextRotaryEmbedding

from export_glimmer import GPTQ_BLOCK, PREFIX, REVISION, SOURCE, SRC
from onnx_export import Weights


def dequant(W: Weights, p: str) -> torch.Tensor:
    packed = W(f'{p}.weight_packed', native=True).numpy().view(np.uint8)
    n, k = (int(v) for v in W(f'{p}.weight_shape', native=True).tolist())
    codes = np.stack([packed & 15, packed >> 4], -1).reshape(n, k).astype(np.float32) - 8
    scale = W(f'{p}.weight_scale', native=True).float().numpy().astype(np.float16).astype(np.float32)
    return torch.from_numpy((codes.reshape(n, k // GPTQ_BLOCK, GPTQ_BLOCK) * scale[..., None]).reshape(n, k))


def rtn4(w: torch.Tensor, block: int = 32) -> torch.Tensor:
    """The embedding as the export rounds it: codes from the float32 scale, read with its float16 copy."""
    g = w.float().reshape(w.shape[0], -1, block)
    s = g.abs().amax(-1, keepdim=True) / 7
    q = torch.clamp(torch.round(g / torch.where(s == 0, 1, s)), -8, 7)
    return (q * s.half().float()).reshape(w.shape)


def nbits4(w: torch.Tensor) -> torch.Tensor:
    """The output layer as onnx_export.matmul_nbits stores it (4 bits, block 32), dequantized."""
    from onnxruntime_genai.models.quantization.cuda_quantizer import CudaQuantizer
    q, s = CudaQuantizer.matmulnbits_blockwise_quantize(w.float(), 4, 32, flatten_qweight=False)
    q = q.numpy().astype(np.uint8).reshape(w.shape[0], -1)
    codes = np.stack([q & 15, q >> 4], -1).reshape(w.shape[0], -1)[:, : w.shape[1]].astype(np.float32) - 8
    scale = s.numpy().astype(np.float16).astype(np.float32).reshape(w.shape[0], -1)
    return torch.from_numpy((codes.reshape(w.shape[0], -1, 32) * scale[..., None]).reshape(w.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_dir', type=Path)
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--tokens', type=int, default=2100)
    parser.add_argument('--decode', type=int, default=3)
    args = parser.parse_args()
    raw = json.loads((SRC / 'config.json').read_text())['text_config']
    cfg = MuseGlimmerTextConfig(**raw)  # all 52 layer types; only the first N layers are built
    cfg._attn_implementation = 'eager'
    W = Weights(SRC, SOURCE, REVISION)
    rotary = MuseGlimmerTextRotaryEmbedding(cfg)
    eps = raw['rms_norm_eps']

    table = W(f'{PREFIX}embed_tokens.weight', native=True)
    head = W('lm_head.weight', native=True)
    final_norm = W(f'{PREFIX}norm.weight')

    def reference(ids: list[int]) -> np.ndarray:
        """Logits of the last token after the whole sequence (recomputed each time: no cache)."""
        x = rtn4(table[torch.tensor(ids)])
        x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
        x = x[None]
        S = x.shape[1]
        pos = torch.arange(S)[None]
        pe = rotary(x, pos)
        q, k = torch.arange(S)[:, None], torch.arange(S)[None, :]
        causal = torch.where(k <= q, 0.0, float('-inf'))[None, None]
        sliding = torch.where((k <= q) & (q - k < raw['sliding_window']), 0.0, float('-inf'))[None, None]
        for i in range(args.layers):
            layer = MuseGlimmerTextDecoderLayer(cfg, i).float().eval()
            p = f'{PREFIX}layers.{i}'
            state = {}
            for name, _ in layer.named_parameters():
                mod = name.rsplit('.', 1)[0]
                state[name] = dequant(W, f'{p}.{mod}') if name.endswith('proj.weight') else W(f'{p}.{name}')
            layer.load_state_dict(state)
            local = raw['layer_types'][i] == 'sliding_attention'
            with torch.no_grad():
                x = layer(x, position_embeddings=pe if raw['layer_rope_theta'][i] else None, attention_mask=sliding if local else causal)
            print(f'  reference layer {i}: max |x| {x.abs().max():.1f}', flush=True)
            del layer, state
            gc.collect()
        h = x[0, -1:]
        h = h * torch.pow(h.pow(2).mean(-1, keepdim=True) + eps, -0.5) * final_norm
        logits = torch.cat([h @ nbits4(head[s:s + 16384]).T for s in range(0, head.shape[0], 16384)], -1)
        cap = raw['final_logit_softcapping']
        return (cap * torch.tanh(logits * raw['output_multiplier'] / cap))[0].numpy()

    session = ort.InferenceSession(str(args.onnx_dir / 'model.onnx'), providers=['CPUExecutionProvider'])
    past = {i.name: np.zeros((1, i.shape[1], 0, i.shape[3]), np.float16) for i in session.get_inputs() if i.name.startswith('past_key_values')}
    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 200000, args.tokens).tolist()
    seen: list[int] = []
    for step, chunk in enumerate([ids] + [[int(t)] for t in rng.integers(1000, 200000, args.decode)]):
        seen += chunk
        want = reference(seen)
        out = session.run(None, {'input_ids': np.array([chunk], np.int64), 'attention_mask': np.ones((1, len(seen)), np.int64), **past})
        names = [o.name for o in session.get_outputs()]
        result = dict(zip(names, out))
        past = {n.replace('present', 'past_key_values'): result[n] for n in names if n.startswith('present')}
        cap = raw['final_logit_softcapping']
        got = cap * np.tanh(result['logits'][0, -1].astype(np.float32) * raw['output_multiplier'] / cap)  # as the app applies it
        cos = float(got @ want / np.linalg.norm(got) / np.linalg.norm(want))
        print(f'step {step} ({len(chunk)} tokens, {len(seen)} seen): cosine {cos:.5f}, same top token {got.argmax() == want.argmax()}, '
              f'max |diff| {np.abs(got - want).max():.3f}', flush=True)


if __name__ == '__main__':
    main()
