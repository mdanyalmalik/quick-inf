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
  (no model names = all models; default precision: the FIXTURES entry's, else q4f16; --tokenizer-only skips the model run)
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_model import MODELS, cache_dir  # noqa: E402

# Fixture file prefix and extra chat-template arguments, per catalogue model.
CHATML_STOPS = ["<|im_end|>", "<|endoftext|>"]
LLAMA3_STOPS = ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]
PHI_STOPS = ["<|end|>", "<|endoftext|>"]
NO_THINKING = {"enable_thinking": False}
# Llama's template writes today's date; a fixed one keeps the fixtures stable (the app passes it too).
LLAMA_DATE = {"date_string": "26 Jul 2024"}
# The day templates that read the clock see (and that fills Mistral's date placeholders).
FIXED_DAY = datetime(2026, 10, 12)
FIXTURES = {
    "smollm2-360m-instruct": {"name": "smollm2", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "qwen2.5-0.5b-instruct": {"name": "qwen2.5-0.5b", "template_kwargs": {}, "stops": CHATML_STOPS},
    "qwen3-0.6b": {"name": "qwen3-0.6b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS},
    "smollm2-1.7b-instruct": {"name": "smollm2-1.7b", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "qwen2.5-1.5b-instruct": {"name": "qwen2.5-1.5b", "template_kwargs": {}, "stops": CHATML_STOPS},
    "qwen2.5-coder-1.5b-instruct": {"name": "qwen2.5-coder-1.5b", "template_kwargs": {}, "stops": CHATML_STOPS},
    "qwen3-1.7b": {"name": "qwen3-1.7b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS},
    "qwen3-4b": {"name": "qwen3-4b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS},
    "llama-3.2-1b-instruct": {"name": "llama-3.2-1b", "template_kwargs": LLAMA_DATE, "stops": LLAMA3_STOPS},
    "llama-3.2-3b-instruct": {"name": "llama-3.2-3b", "template_kwargs": LLAMA_DATE, "stops": LLAMA3_STOPS},
    "phi-4-mini-instruct": {"name": "phi-4-mini", "template_kwargs": {}, "stops": PHI_STOPS},
    "phi-3.5-mini-instruct": {"name": "phi-3.5-mini", "template_kwargs": {}, "stops": PHI_STOPS},
    "qwen3.5-0.8b": {"name": "qwen3.5-0.8b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS, "layout": "qwen3.5"},
    "qwen3.5-2b": {"name": "qwen3.5-2b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS, "layout": "qwen3.5"},
    "qwen3.5-4b": {"name": "qwen3.5-4b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS, "layout": "qwen3.5"},
    # Ternary Qwen3.5 (q2f16 only); its export carries no chat template, so Qwen3.5's (same tokenizer) is used.
    "ternary-bonsai-2-2b": {"name": "ternary-bonsai-2-2b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS, "layout": "qwen3.5",
                            "precision": "q2f16", "template_from": "qwen3.5-0.8b"},
    "deepseek-r1-distill-qwen-1.5b": {"name": "deepseek-r1-qwen-1.5b", "template_kwargs": {}, "stops": ["<｜end▁of▁sentence｜>"]},
    "qwen3-4b-instruct-2507": {"name": "qwen3-4b-2507", "template_kwargs": {}, "stops": CHATML_STOPS},
    "gemma-3-4b-it": {"name": "gemma-3-4b", "template_kwargs": {}, "stops": ["<end_of_turn>", "<eos>"]},
    "smollm2-135m-instruct": {"name": "smollm2-135m", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "gemma-3-270m-it": {"name": "gemma-3-270m", "template_kwargs": {}, "stops": ["<end_of_turn>", "<eos>"]},
    "lfm2.5-350m": {"name": "lfm2.5-350m", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "granite-4.0-350m": {"name": "granite-4.0-350m", "template_kwargs": {}, "stops": ["<|end_of_text|>"]},
    "qwen2.5-coder-0.5b-instruct": {"name": "qwen2.5-coder-0.5b", "template_kwargs": {}, "stops": CHATML_STOPS},
    "lfm2-700m": {"name": "lfm2-700m", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "gemma-3-1b-it": {"name": "gemma-3-1b", "template_kwargs": {}, "stops": ["<end_of_turn>", "<eos>"]},
    "granite-4.0-1b": {"name": "granite-4.0-1b", "template_kwargs": {}, "stops": ["<|end_of_text|>"]},
    "lfm2.5-1.2b-instruct": {"name": "lfm2.5-1.2b", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "qwen2.5-math-1.5b-instruct": {"name": "qwen2.5-math-1.5b", "template_kwargs": {}, "stops": CHATML_STOPS},
    "lfm2-2.6b": {"name": "lfm2-2.6b", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "qwen2.5-coder-3b-instruct": {"name": "qwen2.5-coder-3b", "template_kwargs": {}, "stops": CHATML_STOPS},
    "granite-4.0-micro": {"name": "granite-4.0-micro", "template_kwargs": {}, "stops": ["<|end_of_text|>"]},
    "phi-3-mini-4k-instruct": {"name": "phi-3-mini", "template_kwargs": {}, "stops": PHI_STOPS},
    "olmo-3-7b-instruct": {"name": "olmo-3-7b", "template_kwargs": {"tools": None}, "stops": CHATML_STOPS},
    "lfm2-8b-a1b": {"name": "lfm2-8b-a1b", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "lfm2-24b-a2b": {"name": "lfm2-24b-a2b", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "lfm2.5-24b-a2b": {"name": "lfm2.5-24b-a2b", "template_kwargs": {}, "stops": ["<|im_end|>"]},
    "phi-4-mini-reasoning": {"name": "phi-4-mini-reasoning", "template_kwargs": {}, "stops": ["<|end|>", "<|endoftext|>"]},
    "mistral-7b-instruct-v0.2": {"name": "mistral-7b-v0.2", "template_kwargs": {}, "stops": ["</s>"]},
    "llama-3.1-8b-instruct": {"name": "llama-3.1-8b", "template_kwargs": LLAMA_DATE, "stops": LLAMA3_STOPS},
    "deepseek-r1-distill-llama-8b": {"name": "deepseek-r1-llama-8b", "template_kwargs": {}, "stops": ["<｜end▁of▁sentence｜>"]},
    "gemma-2-9b-it": {"name": "gemma-2-9b", "template_kwargs": {}, "stops": ["<end_of_turn>", "<eos>"]},
    "phi-3-medium-4k-instruct": {"name": "phi-3-medium", "template_kwargs": {}, "stops": PHI_STOPS},
    "phi-4": {"name": "phi-4", "template_kwargs": {}, "stops": ["<|im_end|>", "<|endoftext|>"]},
    "phi-4-reasoning": {"name": "phi-4-reasoning", "template_kwargs": {}, "stops": ["<|im_end|>", "<|endoftext|>"]},
    "tinyllama-1.1b-chat": {"name": "tinyllama-1.1b", "template_kwargs": {}, "stops": ["</s>"]},
    "qwen3-8b": {"name": "qwen3-8b", "template_kwargs": {"enable_thinking": False}, "stops": CHATML_STOPS},
    "qwen3-14b": {"name": "qwen3-14b", "template_kwargs": {"enable_thinking": False}, "stops": CHATML_STOPS},
    "deepseek-r1-distill-qwen-14b": {"name": "deepseek-r1-qwen-14b", "template_kwargs": {}, "stops": ["<｜end▁of▁sentence｜>"]},
    "ministral-3-3b": {"name": "ministral-3-3b", "template_kwargs": {"date_string": "2026-10-12"}, "stops": ["</s>"]},
    "ministral-3-8b": {"name": "ministral-3-8b", "template_kwargs": {"date_string": "2026-10-12"}, "stops": ["</s>"]},
    "smollm3-3b": {"name": "smollm3-3b", "template_kwargs": {"enable_thinking": False, "date_string": "12 October 2026"}, "stops": ["<|im_end|>"]},
    "gemma-3n-e2b-it": {"name": "gemma-3n-e2b", "template_kwargs": {}, "stops": ["<end_of_turn>", "<eos>"]},
    "gemma-4-e2b-it": {"name": "gemma-4-e2b", "template_kwargs": {}, "stops": ["<turn|>", "<eos>"]},
    "gemma-4-e4b-it": {"name": "gemma-4-e4b", "template_kwargs": {}, "stops": ["<turn|>", "<eos>"]},
    "gpt-oss-20b": {"name": "gpt-oss-20b", "template_kwargs": {"date_string": "2026-10-12"}, "stops": ["<|return|>", "<|call|>"]},
    # P3 models (2026-10-12)
    "qwen3.5-9b": {"name": "qwen3.5-9b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS},
    "qwen3.8-27b": {"name": "qwen3.8-27b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS},
    "qwen3.6-35b-a3b": {"name": "qwen3.6-35b-a3b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS},
    "mistral-nemo-12b": {"name": "mistral-nemo-12b", "template_kwargs": {}, "stops": ["</s>"]},
    "granite-4.1-8b": {"name": "granite-4.1-8b", "template_kwargs": {}, "stops": ["<|end_of_text|>"]},
    "mistral-small-24b": {"name": "mistral-small-24b", "template_kwargs": {"date_string": "2026-10-12"}, "stops": ["</s>"]},
    "gemma-4-26b-a4b-it": {"name": "gemma-4-26b-a4b", "template_kwargs": {}, "stops": ["<turn|>", "<eos>"]},
    "gemma-4-12b-it": {"name": "gemma-4-12b", "template_kwargs": {}, "stops": ["<turn|>", "<eos>"]},
    # end P3 models (2026-10-12)
    # own exports (2026-10-13)
    "qwen2.5-7b-instruct": {"name": "qwen2.5-7b", "template_kwargs": {}, "stops": ["<|im_end|>", "<|endoftext|>"]},
    "qwen2.5-coder-7b-instruct": {"name": "qwen2.5-coder-7b", "template_kwargs": {}, "stops": ["<|im_end|>", "<|endoftext|>"]},
    "deepseek-r1-distill-qwen-7b": {"name": "deepseek-r1-qwen-7b", "template_kwargs": {}, "stops": ["<｜end▁of▁sentence｜>"]},
    "deepseek-r1-0528-qwen3-8b": {"name": "deepseek-r1-0528-qwen3-8b", "template_kwargs": {}, "stops": ["<｜end▁of▁sentence｜>"]},
    "glm-4.7-flash": {"name": "glm-4.7-flash", "template_kwargs": {}, "stops": ["<|user|>", "<|endoftext|>", "<|observation|>"]},
    "muse-glimmer-30b": {"name": "muse-glimmer-30b", "template_kwargs": {"date_string": "2026-10-12", "reasoning_strength": "medium"}, "stops": ["<|eot|>"]},
    "nanbeige4.3-33b": {"name": "nanbeige4.3-33b", "template_kwargs": NO_THINKING, "stops": CHATML_STOPS},
    "flint": {"name": "flint", "template_kwargs": {}, "stops": ["<|user|>", "<|end_of_text|>", "<|observation|>", "<|developer|>"]},
    "nemotron-3.5-bolt-e4b": {"name": "nemotron-3.5-bolt-e4b", "template_kwargs": {}, "stops": ["<turn|>", "<eos>"]},
    "gemma-3-12b-it": {"name": "gemma-3-12b", "template_kwargs": {}, "stops": ["<end_of_turn>", "<eos>"]},
    "nemotron-3-nano-30b-a3b": {"name": "nemotron-3-nano", "template_kwargs": {}, "stops": ["<|im_end|>", "</s>"]},
    # end own exports (2026-10-13)
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
        # Templates that read the clock (SmolLM3, gpt-oss) see a fixed day: the app is given the same one.
        env.globals["strftime_now"] = lambda fmt: FIXED_DAY.strftime(fmt)
        # Newer exports keep the template in chat_template.jinja instead of tokenizer_config.json.
        separate = tokenizer_config.parent / "chat_template.jinja"
        source = separate.read_text(encoding="utf-8") if separate.exists() else config["chat_template"]
        # transformers' `{% generation %}` marks (training masks) aren't Jinja; they render nothing.
        source = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", source)
        self.template = env.from_string(source)
        token = lambda key: config.get(key)["content"] if isinstance(config.get(key), dict) else config.get(key)
        self.kwargs = {"bos_token": token("bos_token") or "", "eos_token": token("eos_token") or "", **kwargs}

    def render(self, messages):
        text = self.template.render(messages=messages, add_generation_prompt=True, **self.kwargs)
        # Mistral's default system prompt names the dates as placeholders, which mistral-common fills in (as the app does).
        return text.replace("{today}", FIXED_DAY.strftime("%Y-%m-%d")).replace("{yesterday}", (FIXED_DAY - timedelta(days=1)).strftime("%Y-%m-%d"))


def tokenizer_fixtures(model_id: str, tok: Tokenizer, template: ChatTemplate):
    tok.encode_special_tokens = True  # message text never produces control tokens
    text = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in TEXT_CASES]

    tok.encode_special_tokens = False  # the template's control tokens are real special tokens
    chat = []
    for messages in CHAT_CASES:
        try:
            rendered = template.render(messages)
        except RuntimeError:  # the template refuses it (e.g. no system role); the app merges such turns
            continue
        chat.append({"messages": messages, "rendered": rendered, "ids": tok.encode(rendered, add_special_tokens=False).ids})

    end = tok.token_to_id(FIXTURES[model_id]["stops"][0])
    newline = tok.encode("\n", add_special_tokens=False).ids
    system = tok.encode("system", add_special_tokens=False).ids
    decode = []
    for ids in ([end, *system, *newline, end], text[1]["ids"], text[10]["ids"], text[13]["ids"]):
        decode.append(
            {
                "ids": ids,
                "text": tok.decode(ids, skip_special_tokens=False),
                "textSkipSpecial": tok.decode(ids, skip_special_tokens=True),
            }
        )
    m = MODELS[model_id]
    return {"source": {"repo": m["repo"], "revision": m["revision"], "tokenizers": __import__("tokenizers").__version__,
                       "path": next(p for p in m["files"] if p.endswith("tokenizer.json"))},
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


class Qwen35Runner(Runner):
    """Qwen3.5: token embeddings from their own graph; linear-attention layers carry a
    convolution and a recurrent state instead of a KV cache; position ids have three rows
    (multimodal rotary positions; all equal for text)."""

    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_reuse = False
        embed_path = model_path.parent / model_path.name.replace("decoder_model_merged", "embed_tokens")
        self.embed = ort.InferenceSession(str(embed_path), opts, providers=["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.state_names = [n for n in self.inputs if n.startswith(("past_conv.", "past_recurrent.", "past_key_values."))]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.has_position_ids = True

    def describe(self):
        kv = [n for n in self.state_names if n.startswith("past_key_values.")]
        conv = [n for n in self.state_names if n.startswith("past_conv.")]
        first = self.inputs[kv[0]]
        return {"layout": "qwen3.5", "attentionLayers": len(kv) // 2, "linearLayers": len(conv),
                "kvHeads": first.shape[1], "headDim": first.shape[3],
                "kvType": "float16" if first.type == "tensor(float16)" else "float32", "positionIds": True}

    def empty_past(self):
        past = []
        for name in self.state_names:
            meta = self.inputs[name]
            dtype = np.float16 if meta.type == "tensor(float16)" else np.float32
            shape = [1 if d == "batch_size" else 0 if d == "past_sequence_length" else d for d in meta.shape]
            past.append(np.zeros(shape, dtype=dtype))
        return past

    def step(self, ids, past, past_len):
        n = len(ids)
        embeds = self.embed.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0]
        positions = np.arange(past_len, past_len + n, dtype=np.int64)
        feeds = {
            "inputs_embeds": embeds.astype(np.float32),
            "attention_mask": np.ones((1, past_len + n), dtype=np.int64),
            "position_ids": np.broadcast_to(positions, (3, 1, n)).copy(),
        }
        for name, value in zip(self.state_names, past):
            feeds[name] = value
        outs = dict(zip(self.output_names, self.session.run(None, feeds)))
        logits = outs["logits"][0, -1].astype(np.float32)
        present = [outs[n.replace("past_key_values.", "present.").replace("past_", "present_")] for n in self.state_names]
        return logits, present


def model_fixtures(model_id: str, tok: Tokenizer, template: ChatTemplate, model_path: Path, artifact: str):
    runner = (Qwen35Runner if FIXTURES[model_id].get("layout") == "qwen3.5" else Runner)(model_path)
    print(f"  graph: {runner.describe()}")
    stop_ids = {i for i in (tok.token_to_id(t) for t in FIXTURES[model_id]["stops"]) if i is not None}
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
    precision = None
    if "--precision" in args:
        precision = args[args.index("--precision") + 1]
        args = [a for a in args if a not in ("--precision", precision)]
    names = [a for a in args if not a.startswith("--")] or list(FIXTURES)
    for model_id in names:
        model_precision = precision or FIXTURES[model_id].get("precision", "q4f16")
        print(model_id, model_precision)
        cache = cache_dir(model_id)
        # GenAI builds keep their tokenizer next to the model, in a subfolder.
        tok_path = next(p for p in MODELS[model_id]["files"] if p.endswith("tokenizer.json"))
        tok = Tokenizer.from_file(str(cache / tok_path))
        template_cache = cache_dir(FIXTURES[model_id].get("template_from", model_id))
        template = ChatTemplate(template_cache / tok_path.replace("tokenizer.json", "tokenizer_config.json"), FIXTURES[model_id]["template_kwargs"])
        name = FIXTURES[model_id]["name"]
        if model_precision == FIXTURES[model_id].get("precision", "q4f16"):  # the tokenizer fixture doesn't depend on the weights
            write(ROOT / f"src/tokenizer/__fixtures__/{name}-tokenizer.json", tokenizer_fixtures(model_id, tok, template))
        if "--tokenizer-only" in sys.argv:
            continue
        layout = FIXTURES[model_id].get("layout")
        artifact = f"onnx/decoder_model_merged_{model_precision}.onnx" if layout == "qwen3.5" else f"onnx/model_{model_precision}.onnx"
        write(ROOT / f"src/runtime/__fixtures__/{name}-{model_precision}-reference.json",
              model_fixtures(model_id, tok, template, cache / artifact, artifact))


if __name__ == "__main__":
    main()
