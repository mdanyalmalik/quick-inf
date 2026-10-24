import { findImageModel, runsInApp, type ModelLicense } from '../catalogue/image-models';
import { findSpeechModel } from '../catalogue/speech-models';
import { ImageRecCard } from './recommend/ImageRecommendations';
import { useState, type ReactNode } from 'react';
import type { ModelEntry } from './types';
import type { WorkspaceDefinition } from './workspaces/registry';
import { BackIcon, ChipIcon, PlusIcon, TrashIcon } from './common/icons';
import { deleteAllModels, deleteEverything } from '../store/clear-storage';
import { ModelLoadPanel } from './models/ModelLoadPanel';
import { formatBytes } from './common/format';
import { useStorage } from './runtime/use-runtime';
import { useLocalUsage } from './runtime/use-local-usage';
import { RecommendationCard } from './recommend/RecommendationCard';
import { useRecommendation, useRecommendations } from './recommend/use-recommendations';
import { findModel } from '../catalogue/models';
import { ChatBackupPanel } from './sessions/ChatBackupPanel';

const STEPS = [
  { title: 'Pick a model', text: 'Choose one from the model list.' },
  { title: 'Download it', text: 'It downloads once, then stays saved on this device.' },
  { title: 'Start using it', text: 'Everything runs on your GPU. Your work is saved in this browser.' },
];

/** The start page: what the app is, how to begin, and what's stored on this device. */
export function NoModelSelected({ onBack }: { onBack?: () => void }) {
  return (
    <div className="empty empty--home">
      {onBack && (
        <button className="icon-button empty__back" type="button" onClick={onBack} aria-label="Back to models">
          <BackIcon />
        </button>
      )}
      <div className="empty__icon">
        <ChipIcon width={28} height={28} />
      </div>
      <h1 className="empty__title">Run AI models in your browser</h1>
      <p className="empty__text muted">
        Quick Inference runs AI models directly on this device’s GPU using WebGPU, or on the CPU, more slowly, when there’s no usable
        GPU. There’s no account and no server: what you type and
        create never leaves this device.
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
      <ChatBackupPanel />

      <p className="home-credit muted">
        Made by{' '}
        <a href="https://danyalmalik.net" target="_blank" rel="noopener noreferrer">
          Danyal Malik
        </a>
      </p>
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
              <span>{findModel(id)?.name ?? findImageModel(id)?.name ?? findSpeechModel(id)?.name ?? id}</span>
              <span className="muted">
                {formatBytes(m.savedBytes + m.sharedPartBytes)}
                {m.completeArtifacts.length === 0 && m.partial ? ' · partly downloaded' : ''}
                {m.sharedPartBytes > 0 &&
                  (m.savedBytes === 0 ? ' · a part, kept from sharing it with another device' : ` · ${formatBytes(m.sharedPartBytes)} of it kept from sharing`)}
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
      {storage.available && <ClearStorage modelIds={saved.map(([id]) => id)} />}
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
  const { test } = useRecommendations();
  const imageSpec = model.task === 'image' ? findImageModel(model.id) : undefined;
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
          <dd>{keepTogether(model.parameters)}</dd>
        </div>
        <div>
          <dt>{saved && saved.completeArtifacts.length > 0 ? 'On device' : 'Download'}</dt>
          <dd>{saved && saved.completeArtifacts.length > 0 ? formatBytes(saved.savedBytes) : model.downloadSize}</dd>
        </div>
        <div>
          <dt>{model.extraSpec.label}</dt>
          <dd>{keepTogether(model.extraSpec.value)}</dd>
        </div>
      </dl>

      {/* What to expect here, once the device has been benchmarked. */}
      {test && recommendation && (
        <div className="empty__rec">
          <RecommendationCard rec={recommendation} />
        </div>
      )}
      {test && imageSpec && runsInApp(imageSpec) && (
        <div className="empty__rec">
          <ImageRecCard model={imageSpec} test={test} />
        </div>
      )}

      {model.summary && <p className="empty__summary">{model.summary}</p>}
      {model.license && <LicenseLine license={model.license} />}

      {model.preview ? (
        <p className="empty__preview" role="note">
          <strong>Preview.</strong> {model.name} can’t run in the app yet, so generating makes placeholder images. This shows how working
          with it will look.
        </p>
      ) : (
        <div className="empty__load">
          <ModelLoadPanel modelId={model.id} showReady />
        </div>
      )}

      <button className="button button--primary" type="button" onClick={onNewSession}>
        <PlusIcon />
        New {workspace.sessionLabel.singular}
      </button>

      <UsageLine className="empty__usage" />
    </div>
  );
}

/** A model's licence, with the attribution and use policy it asks for. */
function LicenseLine({ license }: { license: ModelLicense }) {
  return (
    <p className="empty__license muted">
      {license.notice && <strong className="empty__notice">{license.notice}. </strong>}
      Licence:{' '}
      <a href={license.url} target="_blank" rel="noreferrer">
        {license.name}
      </a>
      {license.usePolicyUrl && (
        <>
          {' · '}
          <a href={license.usePolicyUrl} target="_blank" rel="noreferrer">
            Acceptable use policy
          </a>
        </>
      )}
    </p>
  );
}

/** Bottom of "On this device": delete saved models, or everything the app stores, after a confirmation. */
function ClearStorage({ modelIds }: { modelIds: string[] }) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const run = async (label: string, action: () => Promise<void>, reload: boolean) => {
    setBusy(label);
    setError(null);
    try {
      await action();
      if (reload) location.reload();
      else setConfirming(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };
  if (!confirming) {
    return (
      <div className="clear-storage">
        <button type="button" className="button button--ghost button--small clear-storage__open" onClick={() => setConfirming(true)}>
          <TrashIcon width={14} height={14} />
          Clear storage
        </button>
      </div>
    );
  }
  return (
    <div className="clear-storage clear-storage--confirm" role="group" aria-label="Clear storage">
      <p className="clear-storage__question">What should be deleted from this browser?</p>
      <div className="clear-storage__actions">
        <button
          type="button"
          className="button button--ghost button--small"
          disabled={busy !== null || modelIds.length === 0}
          onClick={() => void run('models', () => deleteAllModels(modelIds), false)}
        >
          {busy === 'models' ? 'Deleting…' : 'Delete saved models'}
        </button>
        <button
          type="button"
          className="button button--small clear-storage__danger"
          disabled={busy !== null}
          onClick={() => void run('everything', deleteEverything, true)}
        >
          {busy === 'everything' ? 'Deleting…' : 'Delete everything'}
        </button>
        <button type="button" className="button button--ghost button--small" disabled={busy !== null} onClick={() => setConfirming(false)}>
          Cancel
        </button>
      </div>
      <p className="home-storage__note muted">
        “Delete saved models” keeps your work; models download again when you use them. “Delete everything” also deletes your chats, images
        and settings, and can’t be undone: save a backup first if you want to keep them.
      </p>
      {error && (
        <p className="home-storage__note clear-storage__error" role="alert">
          {error}
        </p>
      )}
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
    <p className={`${className} muted`} title="Model files are measured exactly; your saved work and app data by their saved size.">
      Using {formatBytes(total)} on this device: models {formatBytes(usage.models)} · your work {formatBytes(usage.chats)} · app data{' '}
      {formatBytes(usage.appData)}
    </p>
  );
}

/**
 * A spec value that wraps only between its parts on narrow screens: a bracketed note ("(2.4B active)") is a smaller
 * unit of its own; "512 × 512 to 1024 × 1024" breaks around "to".
 */
function keepTogether(text: string): ReactNode {
  const tight = text.replace(/ × /g, '\u00a0×\u00a0');
  const note = /^(.*?) (\([^)]*\))$/.exec(tight);
  if (!note) return tight;
  return (
    <>
      {note[1]} <span className="spec-grid__note">{note[2]}</span>
    </>
  );
}
