import type { ModelEntry, Session } from '../types';
import type { WorkspaceDefinition } from '../workspaces/registry';
import { PanelCloseIcon, PlusIcon } from '../common/icons';
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
  onClose: () => void;
}

/** Second column: the sessions (chats, for chat models) belonging to one model. */
export function SessionPanel({
  model,
  workspace,
  sessions,
  selectedSessionId,
  onSelectSession,
  onNewSession,
  onClose,
}: SessionPanelProps) {
  const { sessionLabel, SessionIcon } = workspace;
  const groups = groupByRecency(sessions);
  const [width, setWidth] = useResizableWidth('quick-inf:sessions-width', WIDTH);

  return (
    <nav className="sidebar sidebar--sessions" aria-label={sessionLabel.plural} style={{ width }}>
      <div className="sidebar__header">
        <div className="sidebar__heading">
          <span className="sidebar__heading-title">{sessionLabel.plural}</span>
          <span className="sidebar__heading-sub">{model.name}</span>
        </div>
        <button className="icon-button" type="button" onClick={onClose} aria-label={`Close ${sessionLabel.plural.toLowerCase()}`}>
          <PanelCloseIcon />
        </button>
      </div>

      <div className="sidebar__actions">
        <button className="new-button" type="button" onClick={onNewSession}>
          <PlusIcon />
          New {sessionLabel.singular}
        </button>
      </div>

      <div className="sidebar__scroll">
        {groups.length === 0 ? (
          <p className="sidebar__empty muted">No {sessionLabel.plural.toLowerCase()} yet.</p>
        ) : (
          groups.map((group) => (
            <section key={group.label} className="list-group">
              <h2 className="list-group__title">{group.label}</h2>
              <ul className="list">
                {group.items.map((s) => (
                  <li key={s.id}>
                    <button
                      type="button"
                      className="session-item"
                      aria-current={s.id === selectedSessionId ? 'true' : undefined}
                      onClick={() => onSelectSession(s.id)}
                    >
                      <SessionIcon />
                      <span className="session-item__title">{s.title}</span>
                    </button>
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
