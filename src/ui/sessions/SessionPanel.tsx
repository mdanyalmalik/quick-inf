import { useEffect, useState, useSyncExternalStore } from 'react';
import { saveStatus } from '../../conversations/persistence';
import type { ModelEntry, Session } from '../types';
import type { WorkspaceDefinition } from '../workspaces/registry';
import { BackIcon, PanelCloseIcon, PlusIcon, TrashIcon } from '../common/icons';
import { groupByRecency } from './group-by-recency';
import { ResizeHandle } from '../common/ResizeHandle';
import { useResizableWidth, type WidthBounds } from '../common/use-resizable-width';

const WIDTH: WidthBounds = { initial: 260, min: 200, max: 440 };

interface SessionPanelProps {
  model: ModelEntry;
  workspace: WorkspaceDefinition;
  sessions: Session[];
  selectedSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onClose: () => void;
  /** Opens the model's overview page (details, load, new session). */
  onOpenModel: () => void;
  /** Phone layout: the panel fills the screen and "close" means going back to the models. */
  narrow?: boolean;
}

/** Second column: the sessions (chats, for chat models) belonging to one model. */
export function SessionPanel({
  model,
  workspace,
  sessions,
  selectedSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onClose,
  onOpenModel,
  narrow = false,
}: SessionPanelProps) {
  const { sessionLabel, SessionIcon } = workspace;
  const groups = groupByRecency(sessions);
  const [width, setWidth] = useResizableWidth('quick-inf:sessions-width', WIDTH);
  const saveError = useSyncExternalStore(saveStatus.subscribe, saveStatus.getError);

  return (
    <nav className="sidebar sidebar--sessions" aria-label={sessionLabel.plural} style={{ width }}>
      <div className="sidebar__header">
        {narrow && (
          <button className="icon-button" type="button" onClick={onClose} aria-label="Back to models">
            <BackIcon />
          </button>
        )}
        <div className="sidebar__heading">
          <span className="sidebar__heading-title">{sessionLabel.plural}</span>
          <button type="button" className="sidebar__heading-sub" onClick={onOpenModel} title={`${model.name} overview`}>
            {model.name}
          </button>
        </div>
        {!narrow && (
          <button className="icon-button" type="button" onClick={onClose} aria-label={`Close ${sessionLabel.plural.toLowerCase()}`}>
            <PanelCloseIcon />
          </button>
        )}
      </div>

      <div className="sidebar__actions">
        <button className="new-button" type="button" onClick={onNewSession}>
          <PlusIcon />
          New {sessionLabel.singular}
        </button>
      </div>

      {saveError && (
        <p className="sidebar__warning" role="alert" title={saveError}>
          {sessionLabel.plural} aren’t being saved in this browser window. They’ll be lost when it closes.
        </p>
      )}

      <div className="sidebar__scroll">
        {groups.length === 0 ? (
          <p className="sidebar__empty muted">No {sessionLabel.plural.toLowerCase()} yet.</p>
        ) : (
          groups.map((group) => (
            <section key={group.label} className="list-group">
              <h2 className="list-group__title">{group.label}</h2>
              <ul className="list">
                {group.items.map((s) => (
                  <li key={s.id} className="session-row">
                    <button
                      type="button"
                      className="session-item"
                      aria-current={s.id === selectedSessionId ? 'true' : undefined}
                      onClick={() => onSelectSession(s.id)}
                    >
                      <SessionIcon />
                      <span className="session-item__title">{s.title}</span>
                    </button>
                    <DeleteSessionButton
                      label={`${sessionLabel.singular} “${s.title}”`}
                      onDelete={() => onDeleteSession(s.id)}
                    />
                  </li>
                ))}
              </ul>
            </section>
          ))
        )}
      </div>

      <ResizeHandle
        label={`Resize ${sessionLabel.plural.toLowerCase()} panel`}
        width={width}
        bounds={WIDTH}
        onResize={setWidth}
      />
    </nav>
  );
}

const CONFIRM_MS = 4000;

/** Trash button on a session row; the first click asks for confirmation, the second deletes. */
function DeleteSessionButton({ label, onDelete }: { label: string; onDelete: () => void }) {
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!confirming) return;
    const timer = setTimeout(() => setConfirming(false), CONFIRM_MS);
    return () => clearTimeout(timer);
  }, [confirming]);

  return (
    <button
      type="button"
      className={`session-row__delete${confirming ? ' session-row__delete--confirm' : ''}`}
      aria-label={confirming ? `Confirm deleting ${label}` : `Delete ${label}`}
      title={confirming ? 'Click again to delete' : 'Delete'}
      onClick={() => (confirming ? onDelete() : setConfirming(true))}
      onBlur={() => setConfirming(false)}
    >
      {confirming ? 'Delete?' : <TrashIcon width={15} height={15} />}
    </button>
  );
}
