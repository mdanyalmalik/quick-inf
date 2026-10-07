import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import type { WorkspaceProps } from '../registry';
import type { ChatMessage } from './types';
import { placeholderMessagesFor } from '../../placeholder-data';
import { SendIcon } from '../../common/icons';

// Placeholder until the conversation service and runtime adapter exist:
// messages live in component state and replies are canned.
const PLACEHOLDER_REPLY = 'Inference is not connected yet. This is a placeholder response.';

export function ChatWorkspace({ model, session }: WorkspaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(() => placeholderMessagesFor(session.id));
  const [draft, setDraft] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [messages]);

  function send(e?: FormEvent) {
    e?.preventDefault();
    const content = draft.trim();
    if (!content) return;
    const stamp = Date.now();
    setMessages((prev) => [
      ...prev,
      { id: `u${stamp}`, role: 'user', content },
      { id: `a${stamp}`, role: 'assistant', content: PLACEHOLDER_REPLY },
    ]);
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
              <p className="muted">Messages stay on this device.</p>
            </div>
          ) : (
            messages.map((m) => (
              <div key={m.id} className={`message message--${m.role}`}>
                <div className="message__body">{m.content}</div>
              </div>
            ))
          )}
          <div ref={endRef} />
        </div>
      </div>

      <form className="composer" onSubmit={send}>
        <div className="composer__box">
          <textarea
            className="composer__input"
            placeholder={`Message ${model.name}`}
            rows={1}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <button className="composer__send" type="submit" disabled={!draft.trim()} aria-label="Send message">
            <SendIcon />
          </button>
        </div>
        <p className="composer__hint">Enter to send · Shift+Enter for a new line</p>
      </form>
    </div>
  );
}
