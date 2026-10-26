import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { findModel, type ModelSpec } from '../../../catalogue/models';
import { getRuntimeClient } from '../../../runtime/client';
import type { WorkspaceProps } from '../registry';
import type { ChatMessage } from './types';
import { useMessages } from './chat-store';
import { replyThinks, retryReply, sendMessage, stopGeneration, useIsGenerating, useThinking } from './chat-generation';
import { BackIcon, SendIcon, StopIcon } from '../../common/icons';
import { formatMs } from '../../common/format';
import { ModelLoadPanel, UnloadButton } from '../../models/ModelLoadPanel';
import { errorText } from '../../runtime/error-text';
import { modelRuntimeState, useRuntimeStatus } from '../../runtime/use-runtime';
import { AssistantContent } from './MessageContent';

const TITLE_LENGTH = 48;
/** How long a stop may take before offering to force it (a single GPU step is normally milliseconds). */
const FORCE_STOP_AFTER_MS = 4000;

export function ChatWorkspace({ model, session, onRename, onBack }: WorkspaceProps) {
  const { messages, loaded } = useMessages(session.id);
  const generating = useIsGenerating(session.id);
  const status = useRuntimeStatus();
  const runtime = modelRuntimeState(status, model.id);
  const spec = findModel(model.id);
  const [draft, setDraft] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  const ready = runtime === 'ready';
  const canSend = ready && loaded && !generating && draft.trim() !== '' && spec !== undefined;

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [messages]);

  function send(e?: FormEvent) {
    e?.preventDefault();
    if (!canSend || !spec) return;
    const content = draft.trim();
    if (messages.length === 0) onRename(content.length > TITLE_LENGTH ? `${content.slice(0, TITLE_LENGTH)}…` : content);
    sendMessage(session.id, spec, content);
    setDraft('');
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  }

  const retry = spec && ready && !generating ? (replyId: string) => retryReply(session.id, spec, replyId) : undefined;

  return (
    <div className="chat">
      <header className="chat__header">
        {onBack && (
          <button className="icon-button chat__back" type="button" onClick={onBack} aria-label="Back to chats">
            <BackIcon />
          </button>
        )}
        <h1 className="chat__title">{session.title}</h1>
        <span className="chat__model">{model.name}</span>
        {(runtime === 'ready' || runtime === 'busy') && (
          <span className="chat__runtime">
            <span className="status-dot" aria-hidden="true" />
            Loaded
            <UnloadButton disabled={runtime === 'busy'} />
          </span>
        )}
      </header>

      <div className="chat__scroll">
        <div className="chat__messages">
          {!loaded ? null : messages.length === 0 ? (
            <div className="chat__empty">
              <p className="chat__empty-title">Start the conversation</p>
              <p className="muted">Runs on this device; your messages are not sent anywhere.</p>
            </div>
          ) : (
            messages.map((m) => <Message key={m.id} message={m} reasoning={spec ? replyThinks(spec, m) : false} onRetry={retry} />)
          )}
          <div ref={endRef} />
        </div>
      </div>

      <form className="composer" onSubmit={send}>
        <ModelLoadPanel modelId={model.id} />
        <div className="composer__box">
          <textarea
            className="composer__input"
            placeholder={ready || generating ? `Message ${model.name}` : 'Load the model to start chatting'}
            rows={1}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
          />
          {spec?.promptFormat.thinkingSwitch && <ThinkingToggle model={spec} />}
          {generating ? (
            <StopButton sessionId={session.id} stopping={status.state === 'stopping'} />
          ) : (
            <button className="composer__send" type="submit" disabled={!canSend} aria-label="Send message">
              <SendIcon />
            </button>
          )}
        </div>
        <p className="composer__hint">Enter to send · Shift+Enter for a new line</p>
      </form>
    </div>
  );
}

/** Turns thinking on or off for the next replies (models whose template has the switch). */
function ThinkingToggle({ model }: { model: ModelSpec }) {
  const [on, setOn] = useThinking(model);
  return (
    <button
      className="composer__thinking"
      type="button"
      aria-pressed={on}
      title={on ? 'Thinks before answering: slower, often better. Click to answer directly.' : 'Answers directly. Click to think before answering: slower, often better.'}
      onClick={() => setOn(!on)}
    >
      Thinking
    </button>
  );
}

/** Stop button; if stopping stalls, offers a force stop that restarts the runtime. */
function StopButton({ sessionId, stopping }: { sessionId: string; stopping: boolean }) {
  const [slow, setSlow] = useState(false);

  useEffect(() => {
    if (!stopping) {
      setSlow(false);
      return;
    }
    const timer = setTimeout(() => setSlow(true), FORCE_STOP_AFTER_MS);
    return () => clearTimeout(timer);
  }, [stopping]);

  return (
    <>
      {slow && (
        <button
          className="button button--ghost button--small composer__force"
          type="button"
          title="Restarts the runtime immediately. The model will need to be loaded again."
          onClick={() => getRuntimeClient().restart()}
        >
          Force stop
        </button>
      )}
      <button
        className="composer__send composer__send--stop"
        type="button"
        disabled={stopping}
        onClick={() => stopGeneration(sessionId)}
        aria-label={stopping ? 'Stopping' : 'Stop generating'}
      >
        <StopIcon />
      </button>
    </>
  );
}

function Message({ message: m, reasoning, onRetry }: { message: ChatMessage; reasoning: boolean; onRetry?: (replyId: string) => void }) {
  const streaming = m.status === 'streaming';
  return (
    <div className={`message message--${m.role}`}>
      {(m.content || m.role === 'user' || streaming) && (
        <div className="message__body">
          {m.role === 'assistant' ? <AssistantContent text={m.content} reasoning={reasoning} streaming={streaming} /> : m.content}
          {streaming && <span className="cursor" aria-hidden="true" />}
        </div>
      )}
      {m.role === 'assistant' && <MessageMeta message={m} onRetry={onRetry} />}
    </div>
  );
}

const NOT_RETRYABLE = new Set(['context-overflow', 'cancelled']);

function MessageMeta({ message: m, onRetry }: { message: ChatMessage; onRetry?: (replyId: string) => void }) {
  if (m.status === 'error') {
    const code = m.errorCode ?? 'generation-failed';
    const text = errorText(code);
    return (
      <div className="message__error" role="alert">
        <strong>{text.title}</strong>
        {text.hint && <span>{text.hint}</span>}
        {m.error && (
          <details className="error-details">
            <summary>Technical details</summary>
            <code>{m.error}</code>
          </details>
        )}
        {onRetry && !NOT_RETRYABLE.has(code) && (
          <div>
            <button className="button button--ghost button--small" type="button" onClick={() => onRetry(m.id)}>
              Try again
            </button>
          </div>
        )}
      </div>
    );
  }
  if (m.status !== 'done') return null;

  const s = m.stats;
  const notes: string[] = [];
  if (m.interrupted) notes.push('Interrupted: the page closed before this reply finished');
  else if (m.finishReason === 'cancelled') notes.push(s ? 'Stopped' : 'Force stopped; the model was unloaded');
  if (m.finishReason === 'length') notes.push('Reached the length limit');
  if (s && s.droppedTurns > 0) notes.push(`${s.droppedTurns} older messages left out to fit the context`);
  if (notes.length === 0 && !s) return null;

  return (
    <p className="message__meta">
      {notes.map((n) => (
        <span key={n} className="message__note">
          {n}
        </span>
      ))}
      {s && (
        <span>
          {s.generatedTokens} tokens
          {s.decodeTokensPerSecond > 0 && ` · ${s.decodeTokensPerSecond.toFixed(1)} tok/s`}
          {s.timeToFirstTokenMs > 0 && ` · first token ${formatMs(s.timeToFirstTokenMs)}`}
          {` · prompt ${s.promptTokens} tokens`}
        </span>
      )}
    </p>
  );
}

