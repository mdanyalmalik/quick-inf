import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { findModel } from '../../../catalogue/models';
import type { WorkspaceProps } from '../registry';
import type { ChatMessage } from './types';
import { useMessages } from './chat-store';
import { sendMessage, stopGeneration, useIsGenerating } from './chat-generation';
import { SendIcon, StopIcon } from '../../common/icons';
import { formatMs } from '../../common/format';
import { ModelLoadPanel } from '../../models/ModelLoadPanel';
import { modelRuntimeState, useRuntimeStatus } from '../../runtime/use-runtime';

const TITLE_LENGTH = 48;

export function ChatWorkspace({ model, session, onRename }: WorkspaceProps) {
  const messages = useMessages(session.id);
  const generating = useIsGenerating(session.id);
  const runtime = modelRuntimeState(useRuntimeStatus(), model.id);
  const spec = findModel(model.id);
  const [draft, setDraft] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  const ready = runtime === 'ready';
  const canSend = ready && !generating && draft.trim() !== '' && spec !== undefined;

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

  return (
    <div className="chat">
      <header className="chat__header">
        <h1 className="chat__title">{session.title}</h1>
        <span className="chat__model">{model.name}</span>
      </header>

      <div className="chat__scroll">
        <div className="chat__messages">
          {messages.length === 0 ? (
            <div className="chat__empty">
              <p className="chat__empty-title">Start the conversation</p>
              <p className="muted">Runs on this device; your messages are not sent anywhere.</p>
            </div>
          ) : (
            messages.map((m) => <Message key={m.id} message={m} />)
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
          {generating ? (
            <button
              className="composer__send composer__send--stop"
              type="button"
              onClick={() => stopGeneration(session.id)}
              aria-label="Stop generating"
            >
              <StopIcon />
            </button>
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

function Message({ message: m }: { message: ChatMessage }) {
  const streaming = m.status === 'streaming';
  return (
    <div className={`message message--${m.role}`}>
      <div className="message__body">
        {m.content}
        {streaming && <span className="cursor" aria-hidden="true" />}
      </div>
      {m.role === 'assistant' && <MessageMeta message={m} />}
    </div>
  );
}

function MessageMeta({ message: m }: { message: ChatMessage }) {
  if (m.status === 'error') return <p className="message__meta message__meta--error">{m.error}</p>;
  if (m.status !== 'done' || !m.stats) return null;

  const s = m.stats;
  const notes: string[] = [];
  if (m.finishReason === 'cancelled') notes.push('Stopped');
  if (m.finishReason === 'length') notes.push('Reached the length limit');
  if (s.droppedTurns > 0) notes.push(`${s.droppedTurns} older messages left out to fit the context`);

  return (
    <p className="message__meta">
      {notes.map((n) => (
        <span key={n} className="message__note">
          {n}
        </span>
      ))}
      <span>
        {s.generatedTokens} tokens
        {s.decodeTokensPerSecond > 0 && ` · ${s.decodeTokensPerSecond.toFixed(1)} tok/s`}
        {s.timeToFirstTokenMs > 0 && ` · first token ${formatMs(s.timeToFirstTokenMs)}`}
        {` · prompt ${s.promptTokens} tokens`}
      </span>
    </p>
  );
}
