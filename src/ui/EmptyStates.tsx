import type { ModelEntry } from './types';
import type { WorkspaceDefinition } from './workspaces/registry';
import { ChipIcon, PlusIcon } from './common/icons';

export function NoModelSelected() {
  return (
    <div className="empty">
      <div className="empty__icon">
        <ChipIcon width={28} height={28} />
      </div>
      <h1 className="empty__title">Pick a model to get started</h1>
      <p className="empty__text muted">Models run entirely in your browser. Nothing you type leaves this device.</p>
    </div>
  );
}

interface NoSessionSelectedProps {
  model: ModelEntry;
  workspace: WorkspaceDefinition;
  onNewSession: () => void;
}

export function NoSessionSelected({ model, workspace, onNewSession }: NoSessionSelectedProps) {
  return (
    <div className="empty">
      <h1 className="empty__title">{model.name}</h1>
      <p className="empty__text muted">by {model.publisher}</p>

      <dl className="spec-grid">
        <div>
          <dt>Parameters</dt>
          <dd>{model.parameters}</dd>
        </div>
        <div>
          <dt>Download</dt>
          <dd>{model.downloadSize}</dd>
        </div>
        <div>
          <dt>Context</dt>
          <dd>{model.contextLength}</dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd>{model.availability === 'on-device' ? 'On this device' : 'Not downloaded'}</dd>
        </div>
      </dl>

      <button className="button button--primary" type="button" onClick={onNewSession}>
        <PlusIcon />
        New {workspace.sessionLabel.singular}
      </button>
    </div>
  );
}
