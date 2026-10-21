// Qwen3-TTS's prompt tokens (qwen_tts, Qwen/Qwen3-TTS): the text in the chat template
// "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", tokenized as Hugging Face does (special tokens
// on their own, the text between them byte-level BPE), then split into the role tokens (the first 3) and the body
// (the text's tokens). Optional voice instruction for VoiceDesign: "<|im_start|>user\n{instruct}<|im_end|>\n".
// Tests compare with the Python tokenizer (qwen3-tts-text.test.ts).

interface Tokenizer {
  encode(text: string): number[];
  specialTokenId(content: string): number;
}

/** The template's role tokens (<|im_start|>, "assistant", "\n") and the text's own tokens. */
export function qwenPrompt(tokenizer: Tokenizer, text: string): { role: number[]; body: number[] } {
  const start = tokenizer.specialTokenId('<|im_start|>');
  const segment = tokenizer.encode(`assistant\n${text}`);
  const role = [start, ...tokenizer.encode('assistant\n')];
  return { role, body: segment.slice(role.length - 1) };
}

/** VoiceDesign's instruction tokens: "<|im_start|>user\n{instruct}<|im_end|>\n". */
export function qwenInstruct(tokenizer: Tokenizer, instruct: string): number[] {
  return [
    tokenizer.specialTokenId('<|im_start|>'),
    ...tokenizer.encode(`user\n${instruct}`),
    tokenizer.specialTokenId('<|im_end|>'),
    ...tokenizer.encode('\n'),
  ];
}
