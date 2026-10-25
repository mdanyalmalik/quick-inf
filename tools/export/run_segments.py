"""Runs an exported decoder too big for this computer's memory, a slice at a time: the graph is cut into segments of
consecutive nodes (in the file's order, which is topological), each saved as its own model that points into the same
external data file, and each step runs them one after another on the CPU, freeing each before the next. Generates a
reply greedily, with each segment's KV cache carried between steps.

    tools/export/.venv/bin/python tools/export/run_segments.py DIR --segments 8 --prompt-ids 128000,... --steps 40
    tools/export/.venv/bin/python tools/export/run_segments.py DIR --chat "What is the capital of France?"
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
from tokenizers import Tokenizer


def cut(model: onnx.ModelProto, count: int, folder: Path, values: dict) -> list[Path]:
    """`count` segments of about equal weight bytes, saved beside the model (sharing its data file). Each segment's
    inputs take their types from `values` (the first step's), filled in as the earlier segments run."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    size = lambda i: next((int(e.value) for e in i.external_data if e.key == 'length'), len(i.raw_data))
    weight = [sum(size(inits[n]) for n in node.input if n in inits) for node in g.node]
    total, bounds, acc = sum(weight), [0], 0
    for k, w in enumerate(weight):
        acc += w
        if acc >= total * len(bounds) / count and len(bounds) < count:
            bounds.append(k + 1)
    bounds.append(len(g.node))
    graph_inputs = {i.name: i for i in g.input}
    graph_outputs = {o.name for o in g.output}
    for s in range(len(bounds) - 1):
        nodes = g.node[bounds[s]:bounds[s + 1]]
        made = {o for n in nodes for o in n.output if o}
        later = {i for n in g.node[bounds[s + 1]:] for i in n.input}
        read = {i for n in nodes for i in n.input if i}
        ins = [graph_inputs[n] if n in graph_inputs else helper.make_tensor_value_info(n, helper.np_dtype_to_tensor_dtype(values[n].dtype), None)
               for n in sorted(read - made - set(inits)) if n]
        outs = [helper.make_empty_tensor_value_info(n) for n in sorted(made & (later | graph_outputs))]
        sub = helper.make_graph(nodes, f'segment{s}', ins, outs, [inits[n] for n in sorted(read & set(inits))])
        m = helper.make_model(sub, opset_imports=model.opset_import, ir_version=model.ir_version)
        path = folder / f'segment{s}.onnx'
        onnx.save(m, str(path))
        yield path


# A one-turn chat in each model's own format (its chat template's output, without a system turn).
PROMPTS = {
    'llama-nemotron': ('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>'
                       '<|start_header_id|>user<|end_header_id|>\n\n{chat}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'),
    'muse-glimmer': ('<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n'
                     'Current date: 2026-10-25.\n\nReasoning strength: low.\n\n# Valid recipients: "self", "user".<|eot|>'
                     '<|start|>user<|message|>{chat}<|eot|><|start|>assistant'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path)
    parser.add_argument('--segments', type=int, default=10)
    parser.add_argument('--chat', default='What is the capital of France? Answer in one sentence.')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--format', choices=sorted(PROMPTS), default='llama-nemotron')
    parser.add_argument('--stops', default='128001,128008,128009', help='token ids that end the reply')
    args = parser.parse_args()
    model = onnx.load(str(args.dir / 'model.onnx'), load_external_data=False)
    paths = []
    tok = Tokenizer.from_file(str(args.dir / 'tokenizer.json'))
    prompt = PROMPTS[args.format].format(chat=args.chat)
    ids = tok.encode(prompt, add_special_tokens=False).ids
    past = {i.name: np.zeros((1, i.type.tensor_type.shape.dim[1].dim_value, 0, i.type.tensor_type.shape.dim[3].dim_value), np.float16)
            for i in model.graph.input if i.name.startswith('past_key_values')}
    total, chunk, out = 0, ids, []
    options = ort.SessionOptions()
    options.log_severity_level = 3
    for step in range(args.steps):
        total += len(chunk)
        values = {'input_ids': np.array([chunk], np.int64), 'attention_mask': np.ones((1, total), np.int64), **past}
        # The first step writes the segments as it reaches them (their input types come from its values).
        for path in cut(model, args.segments, args.dir, values) if step == 0 else list(paths):
            if step == 0:
                paths.append(path)
            session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
            names = [o.name for o in session.get_outputs()]
            results = session.run(names, {i.name: values[i.name] for i in session.get_inputs()})
            values.update(zip(names, results))
            del session
            gc.collect()
        past = {n.replace('present', 'past_key_values'): values[n] for n in values if n.startswith('present')}
        logits = values['logits'][0, -1].astype(np.float32)
        token = int(logits.argmax())
        out.append(token)
        print(f'step {step}: {token} {tok.decode([token])!r}  top {np.sort(logits)[-3:][::-1].round(2).tolist()}', flush=True)
        if token in {int(t) for t in args.stops.split(',')}:
            break
        chunk = [token]
    print('reply:', tok.decode(out), flush=True)
    for path in paths:
        path.unlink()


if __name__ == '__main__':
    main()
