import type { ModelEntry } from './types';
import type { WorkspaceDefinition } from './workspaces/registry';
import { ChipIcon, PlusIcon } from './common/icons';
import { ModelLoadPanel } from './models/ModelLoadPanel';

export function NoModelSelected() {
  return (
    <div className="empty">
      <div className="empty__icon">
        <ChipIcon width={28} height={28} />
      </div>
      <h1 className="empty__title">Pick a model to get started</h1>
      <p className="empty__text muted">Models run in your browser on this device’s GPU. Your messages are not sent anywhere.</p>
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
      </dl>

      <div className="empty__load">
        <ModelLoadPanel modelId={model.id} showReady />
      </div>

      <button className="button button--primary" type="button" onClick={onNewSession}>
        <PlusIcon />
        New {workspace.sessionLabel.singular}
      </button>
    </div>
  );
}
