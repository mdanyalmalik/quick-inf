"""Generate reference fixtures for the tokenizer and the ONNX generation loop.

Independent reference: Hugging Face `tokenizers` for tokenization, each model's
own Jinja chat template (tokenizer_config.json) for prompt formatting, and ONNX
Runtime (CPU) for the model, using the same pinned files the app downloads.

Outputs (committed), per model:
  src/tokenizer/__fixtures__/<name>-tokenizer.json
  src/runtime/__fixtures__/<name>-<precision>-reference.json

Usage:
  tools/reference/.venv/bin/python -I tools/reference/fetch_model.py [model ...] [--with-q4]
  tools/reference/.venv/bin/python -I tools/reference/make_fixtures.py [model ...] [--precision q4]
  (no model names = all models; default precision q4f16)
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_model import MODELS, cache_dir  # noqa: E402

# Fixture file prefix and extra chat-template arguments, per catalogue model.
FIXTURES = {
    "smollm2-360m-instruct": {"name": "smollm2", "template_kwargs": {}},
    "qwen2.5-0.5b-instruct": {"name": "qwen2.5-0.5b", "template_kwargs": {}},
    "qwen3-0.6b": {"name": "qwen3-0.6b", "template_kwargs": {"enable_thinking": False}},
    "smollm2-1.7b-instruct": {"name": "smollm2-1.7b", "template_kwargs": {}},
}
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
    "CAPS and MiXeD case WORDS; I'M SURE IT'S FINE",
    "def add(a, b):\n    return a + b  # comment\n",
    "const x = { key: [1, 2, 3] }; // JS",
    "Emoji: 😀 🚀 👩‍💻 🇬🇧",
    "Accents: café naïve résumé Ångström",
    "Decomposed accents: café Å",
    "中文分词测试，日本語のテキスト、한국어 텍스트",
    "Arabic: مرحبا بالعالم and Hebrew: שלום",
    "Math: ∑ x² ≤ ∞ → ∫ f(x) dx",
    "Special-looking text <|im_start|> and <|im_end|> <think> should be plain text",
    "URL https://example.com/path?q=1&r=two#frag",
    "a" * 50,
    "Zero​width and non breaking",
    "\n",
    " ",
    "end with space ",
    "trailing newlines\n\n\n",
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


class ChatTemplate:
    """Renders the model's own Jinja chat template, configured like Hugging Face transformers."""

    def __init__(self, tokenizer_config: Path, kwargs: dict):
        config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise RuntimeError(message)

        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(config["chat_template"])
        self.kwargs = kwargs

    def render(self, messages):
        return self.template.render(messages=messages, add_generation_prompt=True, **self.kwargs)


def tokenizer_fixtures(model_id: str, tok: Tokenizer, template: ChatTemplate):
    tok.encode_special_tokens = True  # message text never produces control tokens
    text = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in TEXT_CASES]

    tok.encode_special_tokens = False  # the template's control tokens are real special tokens
    chat = []
    for messages in CHAT_CASES:
        rendered = template.render(messages)
        chat.append({"messages": messages, "rendered": rendered, "ids": tok.encode(rendered, add_special_tokens=False).ids})

    start, end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    newline = tok.encode("\n", add_special_tokens=False).ids
    system = tok.encode("system", add_special_tokens=False).ids
    decode = []
    for ids in ([start, *system, *newline, end], text[1]["ids"], text[10]["ids"], text[13]["ids"]):
        decode.append(
            {
                "ids": ids,
                "text": tok.decode(ids, skip_special_tokens=False),
                "textSkipSpecial": tok.decode(ids, skip_special_tokens=True),
            }
        )
    m = MODELS[model_id]
    return {"source": {"repo": m["repo"], "revision": m["revision"], "tokenizers": __import__("tokenizers").__version__},
            "templateKwargs": FIXTURES[model_id]["template_kwargs"], "text": text, "chat": chat, "decode": decode}


class Runner:
    """Greedy decoding with the KV cache, shaped from the graph's own input metadata."""

    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The CPU backend's buffer reuse fails on some fp16 graphs (Qwen2.5 MHA export:
        # "Shape mismatch attempting to re-use buffer"); it doesn't change the results.
        opts.enable_mem_reuse = False
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        inputs = {i.name: i for i in self.session.get_inputs()}
        self.has_position_ids = "position_ids" in inputs
        self.past_names = sorted(
            (n for n in inputs if n.startswith("past_key_values.")),
            key=lambda n: (int(n.split(".")[1]), n.endswith(".value")),
        )
        first = inputs[self.past_names[0]]
        self.kv_heads, self.head_dim = first.shape[1], first.shape[3]
        self.kv_dtype = np.float16 if first.type == "tensor(float16)" else np.float32
        self.output_names = [o.name for o in self.session.get_outputs()]

    def describe(self):
        return {"layers": len(self.past_names) // 2, "kvHeads": self.kv_heads, "headDim": self.head_dim,
                "kvType": "float16" if self.kv_dtype == np.float16 else "float32",
                "positionIds": self.has_position_ids}

    def step(self, ids, past, past_len):
        n = len(ids)
        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, past_len + n), dtype=np.int64),
        }
        if self.has_position_ids:
            feeds["position_ids"] = np.arange(past_len, past_len + n, dtype=np.int64)[None, :]
        for name, value in zip(self.past_names, past):
            feeds[name] = value
        outs = dict(zip(self.output_names, self.session.run(None, feeds)))
        logits = outs["logits"][0, -1].astype(np.float32)
        present = [outs[n.replace("past_key_values.", "present.")] for n in self.past_names]
        return logits, present

    def empty_past(self):
        return [np.zeros((1, self.kv_heads, 0, self.head_dim), dtype=self.kv_dtype) for _ in self.past_names]

    def greedy(self, prompt_ids, steps, stop_ids):
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
            if nxt in stop_ids:
                break
            ids = [nxt]
        return tokens, top


def model_fixtures(model_id: str, tok: Tokenizer, template: ChatTemplate, model_path: Path, artifact: str):
    runner = Runner(model_path)
    print(f"  graph: {runner.describe()}")
    stop_ids = {i for i in (tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")) if i is not None}
    tok.encode_special_tokens = False
    cases = []
    for messages in GREEDY_PROMPTS:
        prompt_ids = tok.encode(template.render(messages), add_special_tokens=False).ids
        tokens, top = runner.greedy(prompt_ids, STEPS, stop_ids)
        cases.append({"messages": messages, "promptIds": prompt_ids, "greedyIds": tokens,
                      "greedyText": tok.decode(tokens, skip_special_tokens=True), "topK": top})
        print(f"  {messages[-1]['content'][:40]!r} -> {tok.decode(tokens, skip_special_tokens=True)[:70]!r}")
    m = MODELS[model_id]
    return {"source": {"repo": m["repo"], "revision": m["revision"], "artifact": artifact,
                       "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider", "topK": TOP_K,
                       "graph": runner.describe()},
            "cases": cases}


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    args = sys.argv[1:]
    precision = "q4f16"
    if "--precision" in args:
        precision = args[args.index("--precision") + 1]
        args = [a for a in args if a not in ("--precision", precision)]
    names = [a for a in args if not a.startswith("--")] or list(FIXTURES)
    for model_id in names:
        print(model_id, precision)
        cache = cache_dir(model_id)
        tok = Tokenizer.from_file(str(cache / "tokenizer.json"))
        template = ChatTemplate(cache / "tokenizer_config.json", FIXTURES[model_id]["template_kwargs"])
        name = FIXTURES[model_id]["name"]
        if precision == "q4f16":  # the tokenizer fixture doesn't depend on the weights
            write(ROOT / f"src/tokenizer/__fixtures__/{name}-tokenizer.json", tokenizer_fixtures(model_id, tok, template))
        artifact = f"onnx/model_{precision}.onnx"
        write(ROOT / f"src/runtime/__fixtures__/{name}-{precision}-reference.json",
              model_fixtures(model_id, tok, template, cache / artifact, artifact))


if __name__ == "__main__":
    main()
