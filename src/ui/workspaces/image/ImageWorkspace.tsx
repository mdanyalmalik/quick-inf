// An image tab: one image per session. Before the first image, the prompt box sits in the middle of the
// page; on generating it glides to the bottom while the image area grows above it, and the image reveals
// step by step (noise clearing into the picture, as a diffusion model refines it). Generating again replaces
// the image. Settings come from the model's own input list, since models differ.
//
// Models that run (SD Turbo) generate through the runtime: the tab shows the load panel until the model is
// loaded, then each step's predicted image (several-step runs), and the finished picture fading in. Preview
// models make placeholders: steps paced by a timer, a picture drawn from the prompt and seed (placeholder-image.ts).

import { useEffect, useRef, useState, type CSSProperties, type FormEvent, type KeyboardEvent } from 'react';
import type { WorkspaceProps } from '../registry';
import { BackIcon, DownloadIcon, ImageIcon, SendIcon, StopIcon } from '../../common/icons';
import type { Session } from '../../types';
import { sessionStore } from '../../../conversations/session-store';
import {
  defaultSettings,
  findImageModel,
  outputSize,
  type ImageInput,
  type ImageModelSpec,
  type ImageSettings,
  runsInApp,
} from '../../../catalogue/image-models';
import { imageStore, useGeneration, useImageUrl, type Generation } from './image-store';
import { recordImageSpeed } from '../../../recommend/image-speed';
import { getRuntimeClient } from '../../../runtime/client';
import { RuntimeError } from '../../../runtime/contract';
import { ModelLoadPanel, UnloadButton } from '../../models/ModelLoadPanel';
import { errorText } from '../../runtime/error-text';
import { modelRuntimeState, useRuntimeStatus } from '../../runtime/use-runtime';
import { drawFrame, placeholderImageUrl } from './placeholder-image';

const TITLE_LENGTH = 48;
/** Placeholder pacing: each step takes this long, a whole run at most RUN_CAP_MS. */
const STEP_MS = 140;
const RUN_CAP_MS = 3000;
const EXAMPLES = [
  'A lighthouse on a cliff at sunset, oil painting',
  'A cosy reading nook with plants, soft morning light',
  'A red fox in fresh snow, photograph',
];

/** Stops the run in progress, per session. */
const running = new Map<string, () => void>();

export function ImageWorkspace({ model, session, onRename, onBack }: WorkspaceProps) {
  const spec = findImageModel(model.id);
  const generation = useGeneration(session.id);
  const [prompt, setPrompt] = useState(() => generation?.prompt ?? '');
  const [settings, setSettings] = useState<ImageSettings>(() =>
    generation ? { ...generation.settings } : spec ? defaultSettings(spec) : {},
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const runtime = modelRuntimeState(useRuntimeStatus(), model.id);
  const busy = generation?.status === 'generating';

  if (!spec) return null;
  /** A model that runs (not a preview): needs to be loaded before generating. */
  const runs = runsInApp(spec);
  const ready = !runs || runtime === 'ready' || runtime === 'busy';

  function generate(e?: FormEvent) {
    e?.preventDefault();
    const text = prompt.trim();
    if (!text || busy || !spec || !ready) return;
    if (!generation) onRename(text.length > TITLE_LENGTH ? `${text.slice(0, TITLE_LENGTH)}…` : text);
    if (runs) startModelGeneration(session.id, model.id, text, settings);
    else startGeneration(session.id, text, settings);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      generate();
    }
  }

  // Before the first image the prompt box is centred; afterwards the image fills the space above it.
  const started = generation !== null;
  return (
    <div className={`image-tab${started ? ' image-tab--started' : ''}`}>
      <header className="chat__header">
        {onBack && (
          <button className="icon-button chat__back" type="button" onClick={onBack} aria-label="Back to images">
            <BackIcon />
          </button>
        )}
        <h1 className="chat__title">{session.title}</h1>
        <span className="chat__model">{model.name}</span>
        {!runs && <span className="badge badge--muted">Preview</span>}
        {runs && (runtime === 'ready' || runtime === 'busy') && (
          <span className="chat__runtime">
            <span className="status-dot" aria-hidden="true" />
            Loaded
            <UnloadButton disabled={runtime === 'busy'} />
          </span>
        )}
      </header>

      <div className="image-tab__stage">
        <div className="image-tab__spacer" />
        <div className="image-tab__canvas">{generation && <ImageView spec={spec} generation={generation} sessionId={session.id} />}</div>

        <form className="image-tab__composer" onSubmit={generate}>
          {!started && (
            <div className="image-tab__intro">
              <p className="chat__empty-title">Describe an image</p>
              <p className="muted">
                {runs
                  ? `Generated on this device’s GPU.${spec.license?.notice ? ` ${spec.license.notice}.` : ''}`
                  : `Placeholder images for now: ${model.name} can’t run in the app yet.`}
              </p>
            </div>
          )}
          {runs && !ready && (
            <div className="image-tab__load">
              <ModelLoadPanel modelId={model.id} />
            </div>
          )}
          {settingsOpen && <SettingsPanel spec={spec} settings={settings} onChange={setSettings} />}
          <div className="composer__box">
            <textarea
              className="composer__input"
              placeholder={ready ? `Describe an image for ${model.name}` : 'Load the model to start generating'}
              rows={1}
              value={prompt}
              autoFocus
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={onKeyDown}
            />
            <button
              type="button"
              className={`button button--ghost button--small image-settings-toggle${settingsOpen ? ' image-settings-toggle--open' : ''}`}
              aria-expanded={settingsOpen}
              onClick={() => setSettingsOpen(!settingsOpen)}
            >
              Settings
            </button>
            {busy ? (
              <button
                className="composer__send composer__send--stop"
                type="button"
                onClick={() => running.get(session.id)?.()}
                aria-label="Stop generating"
              >
                <StopIcon />
              </button>
            ) : (
              <button
                className="composer__send"
                type="submit"
                disabled={!prompt.trim() || !ready}
                aria-label={started ? 'Generate again' : 'Generate'}
              >
                <SendIcon />
              </button>
            )}
          </div>
          <p className="composer__hint">
            {summarise(spec, settings)} · Enter to {started ? 'generate again (replaces this image)' : 'generate'}
          </p>
          {!started && (
            <div className="image-examples">
              {EXAMPLES.map((example) => (
                <button key={example} type="button" className="button button--ghost button--small" onClick={() => setPrompt(example)}>
                  {example}
                </button>
              ))}
            </div>
          )}
        </form>
      </div>
    </div>
  );
}

/** The frame's shape; its CSS fits it inside the image area by width and by height. */
function frameStyle(width: number, height: number): CSSProperties {
  return { aspectRatio: `${width} / ${height}`, '--image-ratio': width / height } as CSSProperties;
}

const PHASE_LABEL = { prompt: 'Reading the prompt', denoising: 'Generating', decoding: 'Finishing the image' } as const;

/** A real model's image: each step's preview while generating, then the saved picture fading in. */
function ModelImageView({ spec, generation: g, sessionId }: { spec: ImageModelSpec; generation: Generation; sessionId: string }) {
  const { width, height } = outputSize(spec, g.settings);
  const url = useImageUrl(sessionId, g);
  const progress = g.progress ? g.progress.step / Math.max(1, g.progress.steps) : 0;
  const shown = g.status === 'done' ? url : g.previewUrl;
  return (
    <figure className={`image-view image-view--${g.status}`} key={g.id}>
      <div className="image-view__frame" style={frameStyle(width, height)}>
        {shown ? (
          <img
            key={shown}
            src={shown}
            alt={g.prompt}
            width={width}
            height={height}
            className={g.status === 'done' ? 'image-view__img image-view__img--final' : 'image-view__img'}
          />
        ) : (
          <div className="image-view__pending" aria-hidden="true" />
        )}
        {g.status === 'generating' && (
          <div className="image-view__progress">
            <span>
              {g.progress?.phase === 'denoising' || (g.progress?.steps ?? 1) > 1
                ? `${PHASE_LABEL[g.progress?.phase ?? 'prompt']} · ${spec.pipeline?.kind === 'janus' ? 'row' : 'step'} ${g.progress?.step ?? 0} of ${g.progress?.steps ?? 1}`
                : `${PHASE_LABEL[g.progress?.phase ?? 'prompt']}…`}
            </span>
            <span className="image-view__bar">
              <span style={{ width: `${(g.progress?.phase === 'decoding' ? 0.95 : progress * 0.9) * 100}%` }} />
            </span>
          </div>
        )}
        {g.status === 'done' && url && (
          <a
            className="icon-button image-view__download"
            href={url}
            download={`image-${g.seed}.png`}
            title="Download"
            aria-label="Download image"
          >
            <DownloadIcon width={15} height={15} />
          </a>
        )}
      </div>
      <figcaption className="muted">
        {g.status === 'failed' && g.error ? <span className="image-view__error">{errorText(g.error.code as never).title}. </span> : null}
        {g.status === 'cancelled' ? 'Stopped before it finished. ' : ''}
        {summarise(spec, g.settings, g.seed >= 0 ? g.seed : undefined)}
        {g.timings ? ` · ${(g.timings.totalMs / 1000).toFixed(1)} s` : ''}
      </figcaption>
    </figure>
  );
}

/** Starts a generation with the loaded model; progress and the picture arrive through the store. */
function startModelGeneration(sessionId: string, modelId: string, prompt: string, settings: ImageSettings): void {
  const steps = Math.max(1, Number(settings.steps ?? 1));
  const generation: Generation = {
    id: crypto.randomUUID(),
    prompt,
    settings: { ...settings },
    seed: -1,
    createdAt: Date.now(),
    status: 'generating',
    real: true,
    progress: { step: 0, steps, phase: 'prompt' },
  };
  imageStore.set(sessionId, generation);
  sessionStore.touch(sessionId);
  const current = () => {
    const g = imageStore.get(sessionId);
    return g?.id === generation.id ? g : null;
  };
  const handle = getRuntimeClient().generateImage({ prompt, settings }, (p) => {
    const g = current();
    if (!g) return;
    imageStore.set(sessionId, {
      ...g,
      progress: { step: p.step, steps: p.steps, phase: p.phase },
      previewUrl: p.preview ? URL.createObjectURL(p.preview) : g.previewUrl,
    });
  });
  running.set(sessionId, handle.cancel);
  handle.result
    .then(async (r) => {
      recordImageSpeed(modelId, steps, settings, r.timings);
      await imageStore.saveImage(sessionId, generation.id, r.png);
      const g = current();
      if (g)
        imageStore.set(sessionId, {
          ...g,
          seed: r.seed,
          status: 'done',
          progress: undefined,
          previewUrl: undefined,
          timings: { totalMs: r.timings.totalMs },
        });
      sessionStore.touch(sessionId);
    })
    .catch((err: unknown) => {
      const g = current();
      if (!g) return;
      const code = err instanceof RuntimeError ? err.code : 'generation-failed';
      const message = err instanceof Error ? err.message : String(err);
      imageStore.set(sessionId, {
        ...g,
        status: code === 'cancelled' ? 'cancelled' : 'failed',
        progress: undefined,
        error: code === 'cancelled' ? undefined : { code, message },
      });
    })
    .finally(() => running.delete(sessionId));
}

function ImageView({ spec, generation, sessionId }: { spec: ImageModelSpec; generation: Generation; sessionId: string }) {
  return generation.real ? (
    <ModelImageView spec={spec} generation={generation} sessionId={sessionId} />
  ) : (
    <PlaceholderImageView spec={spec} generation={generation} />
  );
}

/** A placeholder image: drawn step by step while generating, then the finished picture with a download link. */
function PlaceholderImageView({ spec, generation: g }: { spec: ImageModelSpec; generation: Generation }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width, height } = outputSize(spec, g.settings);
  const progress = g.status === 'generating' ? (g.progress?.step ?? 0) / Math.max(1, g.progress?.steps ?? 1) : 1;

  useEffect(() => {
    if (canvasRef.current)
      drawFrame(canvasRef.current, g.prompt, g.seed, g.status === 'cancelled' ? (g.progress ? progress : 0.15) : progress);
  }, [g.prompt, g.seed, g.status, progress, g.progress]);

  return (
    <figure className={`image-view image-view--${g.status}`} key={g.id}>
      <div className="image-view__frame" style={frameStyle(width, height)}>
        <canvas ref={canvasRef} width={width} height={height} aria-label={g.prompt} role="img" />
        {g.status === 'generating' && (
          <div className="image-view__progress">
            <span>
              Step {g.progress?.step ?? 0} of {g.progress?.steps ?? 0}
            </span>
            <span className="image-view__bar">
              <span style={{ width: `${progress * 100}%` }} />
            </span>
          </div>
        )}
        {g.status === 'done' && (
          <a
            className="icon-button image-view__download"
            href={placeholderImageUrl(g.prompt, g.seed, width, height)}
            download={`image-${g.seed}.png`}
            title="Download"
            aria-label="Download image"
          >
            <DownloadIcon width={15} height={15} />
          </a>
        )}
      </div>
      <figcaption className="muted">
        {g.status === 'cancelled' ? 'Stopped before it finished. ' : ''}
        {summarise(spec, g.settings, g.seed)}
      </figcaption>
    </figure>
  );
}

/** Starts a placeholder run: steps paced by a timer, the picture sharpening as they go. */
function startGeneration(sessionId: string, prompt: string, settings: ImageSettings): void {
  const chosen = Number(settings.seed ?? -1);
  const seed = chosen >= 0 ? chosen : Math.floor(Math.random() * 2 ** 31);
  const steps = Math.max(1, Number(settings.steps ?? 20));
  // A one-step model still shows a short reveal.
  const shown = Math.max(steps, 8);
  const generation: Generation = {
    id: crypto.randomUUID(),
    prompt,
    settings: { ...settings },
    seed,
    createdAt: Date.now(),
    status: 'generating',
    progress: { step: 0, steps },
  };
  imageStore.set(sessionId, generation);
  sessionStore.touch(sessionId);
  const tickMs = Math.min(STEP_MS, RUN_CAP_MS / shown);
  let tick = 0;
  const timer = setInterval(() => {
    tick++;
    const step = Math.min(steps, Math.round((tick / shown) * steps));
    if (tick >= shown) {
      clearInterval(timer);
      running.delete(sessionId);
      imageStore.set(sessionId, { ...generation, status: 'done', progress: undefined });
      sessionStore.touch(sessionId);
    } else {
      imageStore.set(sessionId, { ...generation, progress: { step, steps } });
    }
  }, tickMs);
  running.set(sessionId, () => {
    clearInterval(timer);
    running.delete(sessionId);
    const current = imageStore.get(sessionId);
    imageStore.set(sessionId, { ...generation, status: 'cancelled', progress: current?.progress });
  });
}

/** The model's settings, one control per input it takes. */
function SettingsPanel({
  spec,
  settings,
  onChange,
}: {
  spec: ImageModelSpec;
  settings: ImageSettings;
  onChange: (s: ImageSettings) => void;
}) {
  const set = (id: string, value: string | number) => onChange({ ...settings, [id]: value });
  return (
    <div className="image-settings">
      {spec.inputs.map((input) => (
        <label key={input.id} className={`image-setting image-setting--${input.kind}`} title={input.hint}>
          <span className="image-setting__label">
            {input.label}
            {input.kind === 'range' && <span className="image-setting__value">{settings[input.id]}</span>}
          </span>
          <SettingControl input={input} value={settings[input.id]} onChange={(v) => set(input.id, v)} />
        </label>
      ))}
      <button
        type="button"
        className="button button--ghost button--small image-settings__reset"
        onClick={() => onChange(defaultSettings(spec))}
      >
        Reset
      </button>
    </div>
  );
}

function SettingControl({
  input,
  value,
  onChange,
}: {
  input: ImageInput;
  value: string | number | undefined;
  onChange: (v: string | number) => void;
}) {
  switch (input.kind) {
    case 'text':
      return <input type="text" placeholder={input.placeholder} value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />;
    case 'select':
      return (
        <select value={String(value ?? input.default)} onChange={(e) => onChange(e.target.value)}>
          {input.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      );
    case 'range':
      return (
        <input
          type="range"
          min={input.min}
          max={input.max}
          step={input.step}
          value={Number(value ?? input.default)}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      );
    case 'seed': {
      const seed = Number(value ?? -1);
      return (
        <span className="image-setting__seed">
          <input
            type="number"
            min={0}
            placeholder="Random"
            value={seed < 0 ? '' : seed}
            onChange={(e) => onChange(e.target.value === '' ? -1 : Math.max(0, Math.floor(Number(e.target.value))))}
          />
          <button type="button" className="button button--ghost button--small" onClick={() => onChange(-1)} disabled={seed < 0}>
            Random
          </button>
        </span>
      );
    }
  }
}

/** One line describing the settings, e.g. "512 × 512 · 25 steps · guidance 7.5 · seed 42". */
function summarise(spec: ImageModelSpec, settings: ImageSettings, usedSeed?: number): string {
  const { width, height } = outputSize(spec, settings);
  const parts = [`${width} × ${height}`];
  if (settings.steps !== undefined) parts.push(`${settings.steps} ${Number(settings.steps) === 1 ? 'step' : 'steps'}`);
  if (settings.guidance !== undefined) parts.push(`guidance ${settings.guidance}`);
  if (settings.negative) parts.push('negative prompt');
  if (usedSeed !== undefined) parts.push(`seed ${usedSeed}`);
  else parts.push(Number(settings.seed) >= 0 ? `seed ${settings.seed}` : 'random seed');
  return parts.join(' · ');
}

/** The session's image in the sessions list once it's done; the image icon until then. */
export function ImageThumbnail({ session }: { session: Session }) {
  const generation = useGeneration(session.id);
  const spec = findImageModel(session.modelId);
  const url = useImageUrl(session.id, generation);
  if (!spec || generation?.status !== 'done') return <ImageIcon />;
  if (generation.real) return url ? <img className="session-item__thumb" src={url} alt="" /> : <ImageIcon />;
  const { width, height } = outputSize(spec, generation.settings);
  return <img className="session-item__thumb" src={placeholderImageUrl(generation.prompt, generation.seed, width, height)} alt="" />;
}

/** Stops any run in the session and drops its image (before the session is deleted). */
export function forgetImageSession(sessionId: string): void {
  running.get(sessionId)?.();
  imageStore.forget(sessionId);
}
