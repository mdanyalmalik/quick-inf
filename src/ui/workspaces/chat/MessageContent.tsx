// An assistant reply as formatted text (Markdown: bold, lists, code, tables; maths with KaTeX) with
// a reasoning model's thinking in a collapsible block above the answer.

import { memo } from 'react';
import Markdown, { type Components } from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import { splitReasoning } from '../../../conversations/reasoning';
import { normalizeMath } from './math-text';

// Replies never load anything from the network: images (which would fetch their URL) are
// shown as their alt text, and links open in a new tab without passing this page along.
const COMPONENTS: Components = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer nofollow">
      {children}
    </a>
  ),
};

const Formatted = memo(function Formatted({ text }: { text: string }) {
  return (
    <div className="markdown">
      <Markdown
        remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }]]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={COMPONENTS}
        disallowedElements={['img']}
        unwrapDisallowed
      >
        {normalizeMath(text)}
      </Markdown>
    </div>
  );
});

export function AssistantContent({ text, reasoning, streaming }: { text: string; reasoning: boolean; streaming: boolean }) {
  const { thinking, thinkingOpen, answer } = splitReasoning(text, reasoning);
  const thinkingNow = streaming && thinkingOpen;
  return (
    <>
      {thinking !== null && (
        <details className="thinking" open={thinkingNow || undefined}>
          <summary>{thinkingNow ? 'Thinking…' : `Thought for ${wordCount(thinking)} words`}</summary>
          <div className="thinking__body">
            <Formatted text={thinking} />
          </div>
        </details>
      )}
      {answer && <Formatted text={answer} />}
    </>
  );
}

function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}
