"""Generate reference fixtures for the tokenizer and the ONNX generation loop.

Independent reference: Hugging Face `tokenizers` for tokenization and ONNX
Runtime (CPU) for the model, using the same pinned files the app downloads.

Outputs (committed):
  src/tokenizer/__fixtures__/smollm2-tokenizer.json
  src/runtime/__fixtures__/smollm2-q4f16-reference.json

Usage:
  tools/reference/.venv/bin/python -I tools/reference/fetch_model.py
  tools/reference/.venv/bin/python -I tools/reference/make_fixtures.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_model import CACHE, REPO, REVISION  # noqa: E402

DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
NUM_LAYERS = 32
KV_HEADS = 5
HEAD_DIM = 64
TOP_K = 10
STEPS = 32

TEXT_CASES = [
    "",
    "Hello, world!",
    "hello world",
    "  leading and trailing spaces  ",
    "multiple   spaces\tand\ttabs\n\nand newlines\r\n",
    "Numbers: 0 7 42 1234567890 3.14159 -12,345",
    "I'm sure you're right, they'll see it's done; we've I'd",
    "CAPS and MiXeD case WORDS",
    "def add(a, b):\n    return a + b  # comment\n",
    "const x = { key: [1, 2, 3] }; // JS",
    "Emoji: 😀 🚀 👩‍💻 🇬🇧",
    "Accents: café naïve résumé Ångström",
    "中文分词测试，日本語のテキスト、한국어 텍스트",
    "Arabic: مرحبا بالعالم and Hebrew: שלום",
    "Math: ∑ x² ≤ ∞ → ∫ f(x) dx",
    "Special-looking text <|im_start|> and <|im_end|> should be plain text",
    "URL https://example.com/path?q=1&r=two#frag",
    "a" * 50,
    "Zero​width and non breaking",
    "\n",
    " ",
    "end with space ",
]

CHAT_CASES = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Say hi."},
    ],
    [
        {"role": "user", "content": "Write a haiku about rain."},
        {"role": "assistant", "content": "Soft rain on the roof\nwhispers to the sleeping town\npuddles hold the sky"},
        {"role": "user", "content": "Now one about snow."},
    ],
]

GREEDY_PROMPTS = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Write a Python function that reverses a string."}],
    [{"role": "user", "content": "Give me three tips for better sleep."}],
]


def chatml(messages, add_generation_prompt=True):
    """Mirror of the model's Jinja chat template (tokenizer_config.json)."""
    out = ""
    for i, m in enumerate(messages):
        if i == 0 and m["role"] != "system":
            out += f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out


def tokenizer_fixtures(tok_path: Path):
    tok = Tokenizer.from_file(str(tok_path))
    tok.encode_special_tokens = True  # message text never produces control tokens
    text = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in TEXT_CASES]

    tok.encode_special_tokens = False  # the template's control tokens are real special tokens
    chat = []
    for messages in CHAT_CASES:
        rendered = chatml(messages)
        chat.append({"messages": messages, "rendered": rendered, "ids": tok.encode(rendered, add_special_tokens=False).ids})

    decode = []
    for ids in ([1, 9690, 198, 2], text[1]["ids"], text[10]["ids"], text[12]["ids"]):
        decode.append(
            {
                "ids": ids,
                "text": tok.decode(ids, skip_special_tokens=False),
                "textSkipSpecial": tok.decode(ids, skip_special_tokens=True),
            }
        )
    return {"source": {"repo": REPO, "revision": REVISION, "tokenizers": __import__("tokenizers").__version__},
            "text": text, "chat": chat, "decode": decode}


class Runner:
    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    def step(self, ids, past, past_len):
        n = len(ids)
        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, past_len + n), dtype=np.int64),
            "position_ids": np.arange(past_len, past_len + n, dtype=np.int64)[None, :],
        }
        for i in range(NUM_LAYERS):
            feeds[f"past_key_values.{i}.key"] = past[2 * i]
            feeds[f"past_key_values.{i}.value"] = past[2 * i + 1]
        outs = self.session.run(None, feeds)
        logits = outs[0][0, -1].astype(np.float32)
        return logits, outs[1:]

    @staticmethod
    def empty_past():
        return [np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=np.float16) for _ in range(2 * NUM_LAYERS)]

    def greedy(self, prompt_ids, steps, eos):
        """Greedy decode, recording top-k logits at every step (for teacher-forced comparison)."""
        past, past_len = self.empty_past(), 0
        ids, tokens, top = list(prompt_ids), [], []
        for _ in range(steps):
            logits, past = self.step(ids, past, past_len)
            past_len += len(ids)
            order = np.argsort(-logits)[:TOP_K]
            top.append({"ids": order.tolist(), "logits": [round(float(logits[j]), 4) for j in order]})
            nxt = int(order[0])
            tokens.append(nxt)
            if nxt == eos:
                break
            ids = [nxt]
        return tokens, top


def model_fixtures(tok_path: Path, model_path: Path):
    tok = Tokenizer.from_file(str(tok_path))
    runner = Runner(model_path)
    cases = []
    for messages in GREEDY_PROMPTS:
        prompt_ids = tok.encode(chatml(messages), add_special_tokens=False).ids
        tokens, top = runner.greedy(prompt_ids, STEPS, eos=2)
        cases.append({"messages": messages, "promptIds": prompt_ids, "greedyIds": tokens,
                      "greedyText": tok.decode(tokens, skip_special_tokens=True), "topK": top})
        print(f"  {messages[-1]['content'][:40]!r} -> {tok.decode(tokens, skip_special_tokens=True)[:70]!r}")
    return {"source": {"repo": REPO, "revision": REVISION, "artifact": "onnx/model_q4f16.onnx",
                       "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider", "topK": TOP_K},
            "cases": cases}


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    tok_path = CACHE / "tokenizer.json"
    model_path = CACHE / "onnx" / "model_q4f16.onnx"
    write(ROOT / "src/tokenizer/__fixtures__/smollm2-tokenizer.json", tokenizer_fixtures(tok_path))
    write(ROOT / "src/runtime/__fixtures__/smollm2-q4f16-reference.json", model_fixtures(tok_path, model_path))


if __name__ == "__main__":
    main()
