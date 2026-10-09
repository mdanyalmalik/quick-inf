import type { ModelEntry } from './types';
import type { WorkspaceDefinition } from './workspaces/registry';
import { BackIcon, ChipIcon, PlusIcon } from './common/icons';
import { ModelLoadPanel } from './models/ModelLoadPanel';
import { formatBytes } from './common/format';
import { useStorage } from './runtime/use-runtime';
import { useLocalUsage } from './runtime/use-local-usage';
import { RecommendationCard } from './recommend/RecommendationCard';
import { useRecommendation } from './recommend/use-recommendations';
import { findModel } from '../catalogue/models';

const STEPS = [
  { title: 'Pick a model', text: 'Choose one from the list on the left.' },
  { title: 'Load it', text: 'It downloads once, then stays saved on this device.' },
  { title: 'Start chatting', text: 'Replies run on your GPU. Chats are saved in this browser.' },
];

/** The start page: what the app is, how to begin, and what's stored on this device. */
export function NoModelSelected() {
  return (
    <div className="empty empty--home">
      <div className="empty__icon">
        <ChipIcon width={28} height={28} />
      </div>
      <h1 className="empty__title">Run AI models in your browser</h1>
      <p className="empty__text muted">
        Quick Inference runs language models directly on this device’s GPU using WebGPU. There’s no account and no
        server: your messages never leave this device.
      </p>

      <ol className="home-steps">
        {STEPS.map((step, i) => (
          <li key={step.title} className="home-steps__item">
            <span className="home-steps__number">{i + 1}</span>
            <span className="home-steps__title">{step.title}</span>
            <span className="home-steps__text muted">{step.text}</span>
          </li>
        ))}
      </ol>

      <StoragePanel />
    </div>
  );
}

/** Models saved on this device. (No usage/quota figures: Chrome reports a fixed, not real, quota.) */
function StoragePanel() {
  const storage = useStorage();
  const saved = Object.entries(storage.models);

  return (
    <section className="home-storage" aria-labelledby="home-storage-title">
      <h2 id="home-storage-title" className="home-storage__title">
        On this device
      </h2>
      {!storage.available ? (
        <p className="muted">This browser window can’t save models, so each load downloads the model again.</p>
      ) : saved.length === 0 ? (
        <p className="muted">No models saved yet. A model is saved here the first time you load it.</p>
      ) : (
        <ul className="home-storage__list">
          {saved.map(([id, m]) => (
            <li key={id}>
              <span>{findModel(id)?.name ?? id}</span>
              <span className="muted">
                {formatBytes(m.savedBytes)}
                {m.completeArtifacts.length === 0 && m.partial ? ' · partly downloaded' : ''}
              </span>
            </li>
          ))}
          {storage.otherBytes > 0 && (
            <li>
              <span>Older model files</span>
              <span className="muted">{formatBytes(storage.otherBytes)}</span>
            </li>
          )}
        </ul>
      )}
      <UsageLine className="home-storage__note" />
      {saved.length > 0 && storage.persisted !== undefined && (
        <p className="home-storage__note muted">
          {storage.persisted
            ? 'The browser keeps saved models until you delete them.'
            : 'The browser may clear saved models if your disk runs low on space.'}
        </p>
      )}
    </section>
  );
}

interface NoSessionSelectedProps {
  model: ModelEntry;
  workspace: WorkspaceDefinition;
  onNewSession: () => void;
  /** Phone layout: back to the session list. */
  onBack?: () => void;
}

export function NoSessionSelected({ model, workspace, onNewSession, onBack }: NoSessionSelectedProps) {
  const storage = useStorage();
  const saved = storage.models[model.id];
  const recommendation = useRecommendation(model.id);
  return (
    <div className="empty empty--scroll">
      {onBack && (
        <button className="icon-button empty__back" type="button" onClick={onBack} aria-label="Back">
          <BackIcon />
        </button>
      )}
      <h1 className="empty__title">{model.name}</h1>
      <p className="empty__text muted">by {model.publisher}</p>

      <dl className="spec-grid">
        <div>
          <dt>Parameters</dt>
          <dd>{model.parameters}</dd>
        </div>
        <div>
          <dt>{saved && saved.completeArtifacts.length > 0 ? 'On device' : 'Download'}</dt>
          <dd>{saved && saved.completeArtifacts.length > 0 ? formatBytes(saved.savedBytes) : model.downloadSize}</dd>
        </div>
        <div>
          <dt>Context</dt>
          <dd>{model.contextLength}</dd>
        </div>
      </dl>

      {recommendation && (
        <div className="empty__rec">
          <RecommendationCard rec={recommendation} />
        </div>
      )}

      <div className="empty__load">
        <ModelLoadPanel modelId={model.id} showReady />
      </div>

      <button className="button button--primary" type="button" onClick={onNewSession}>
        <PlusIcon />
        New {workspace.sessionLabel.singular}
      </button>

      <UsageLine className="empty__usage" />
    </div>
  );
}

/**
 * What this app stores on the device: model files (exact), chats and app data
 * (measured as data; the database's own overhead isn't visible to the page).
 */
function UsageLine({ className }: { className: string }) {
  const usage = useLocalUsage();
  if (usage.chats === null) return null;
  const total = usage.models + usage.chats + usage.appData;
  return (
    <p className={`${className} muted`} title="Model files are measured exactly; chats and app data by their saved size.">
      Using {formatBytes(total)} on this device: models {formatBytes(usage.models)} · chats {formatBytes(usage.chats)} · app
      data {formatBytes(usage.appData)}
    </p>
  );
}
