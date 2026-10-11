// Splits a reasoning model's reply into its thinking and its answer.
//
// Thinking models write their reasoning between <think> and </think>, then the
// answer. Some (DeepSeek-R1 distills) leave out the opening tag. The thinking is
// shown apart from the answer and left out of the history the model sees on the
// next turn, as these models' own chat templates do.

export interface ReplyParts {
  /** The reasoning, if any (without its tags). */
  thinking: string | null;
  /** Still thinking: no </think> yet. */
  thinkingOpen: boolean;
  answer: string;
}

const OPEN = '<think>';
const CLOSE = '</think>';

/**
 * `reasoning`: the model always thinks first, so text before any </think> is
 * thinking even without an opening tag.
 */
export function splitReasoning(text: string, reasoning = false): ReplyParts {
  const open = text.indexOf(OPEN);
  const close = text.indexOf(CLOSE);
  // Only a tag at the start counts as opening the thinking (not one quoted in an answer).
  const opensAtStart = open >= 0 && text.slice(0, open).trim() === '';
  if (close >= 0 && (opensAtStart || reasoning)) {
    const start = opensAtStart ? open + OPEN.length : 0;
    return { thinking: text.slice(start, close).trim(), thinkingOpen: false, answer: text.slice(close + CLOSE.length).trimStart() };
  }
  if (opensAtStart) return { thinking: text.slice(open + OPEN.length).trim(), thinkingOpen: true, answer: '' };
  if (reasoning && close < 0) return { thinking: text.trim(), thinkingOpen: true, answer: '' };
  return { thinking: null, thinkingOpen: false, answer: text };
}

/** What goes back to the model as a past reply: the answer alone. */
export function historyText(text: string, reasoning = false): string {
  return splitReasoning(text, reasoning).answer;
}
