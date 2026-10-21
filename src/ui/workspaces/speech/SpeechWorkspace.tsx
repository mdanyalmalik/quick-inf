// A speech tab: type text, hear it spoken. One clip per session; speaking again replaces it. The whole clip is made
// first (with progress shown), then saved as a WAV and played, with a player and a download link. Settings (voice,
// language, speed, steps) come from the model.

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { findSpeechModel, type SpeechModelSpec, type SpeechSettings } from '../../../catalogue/speech-models';
import { sessionStore } from '../../../conversations/session-store';
import { getRuntimeClient } from '../../../runtime/client';
import { RuntimeError } from '../../../runtime/contract';
import { BackIcon, DownloadIcon, SendIcon, StopIcon } from '../../common/icons';
import { ModelLoadPanel, UnloadButton } from '../../models/ModelLoadPanel';
import { errorText } from '../../runtime/error-text';
import { modelRuntimeState, useRuntimeStatus } from '../../runtime/use-runtime';
import type { WorkspaceProps } from '../registry';
import { speechStore, useClip, useClipUrl, type Clip } from './speech-store';
import { encodeWav } from './wav';

const TITLE_LENGTH = 48;
const EXAMPLES = [
  'The quick brown fox jumps over the lazy dog.',
  'After the storm passed, the streets were quiet and the air smelled of rain.',
  'Welcome! This voice was made entirely on your own device.',
];

/** Stops the run in progress, per session. */
const running = new Map<string, () => void>();
/** Clips just finished in this visit, to play once when shown. */
const autoplay = new Set<string>();

/** Whole seconds since `since` (ms), ticking each second; 0 when null. */
function useElapsedSeconds(since: number | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (since === null) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [since]);
  return since === null ? 0 : Math.max(0, Math.floor((now - since) / 1000));
}

export function SpeechWorkspace({ model, session, onRename, onBack }: WorkspaceProps) {
  const spec = findSpeechModel(model.id);
  const clip = useClip(session.id);
  const [text, setText] = useState(() => clip?.text ?? '');
  const [settings, setSettings] = useState<SpeechSettings>(() => (clip ? { ...clip.settings } : { ...spec!.defaults }));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const runtime = modelRuntimeState(useRuntimeStatus(), model.id);
  const busy = clip?.status === 'generating';
  if (!spec) return null;
  const ready = runtime === 'ready' || runtime === 'busy';

  function speak(e?: FormEvent) {
    e?.preventDefault();
    const words = text.trim();
    if (!words || busy || !spec || !ready) return;
    if (!clip) onRename(words.length > TITLE_LENGTH ? `${words.slice(0, TITLE_LENGTH)}…` : words);
    startSpeech(session.id, spec, words, settings);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      speak();
    }
  }

  const started = clip !== null;
  return (
    <div className={`image-tab speech-tab${started ? ' image-tab--started' : ''}`}>
      <header className="chat__header">
        {onBack && (
          <button className="icon-button chat__back" type="button" onClick={onBack} aria-label="Back to clips">
            <BackIcon />
          </button>
        )}
        <h1 className="chat__title">{session.title}</h1>
        <span className="chat__model">{model.name}</span>
        {ready && (
          <span className="chat__runtime">
            <span className="status-dot" aria-hidden="true" />
            Loaded
            <UnloadButton disabled={runtime === 'busy'} />
          </span>
        )}
      </header>

      <div className="image-tab__stage">
        <div className="image-tab__spacer" />
        <div className="image-tab__canvas">{clip && <ClipView spec={spec} clip={clip} sessionId={session.id} />}</div>

        <form className="image-tab__composer" onSubmit={speak}>
          {!started && (
            <div className="image-tab__intro">
              <p className="chat__empty-title">Type something to hear it spoken</p>
              <p className="muted">Spoken on this device; the text never leaves it.</p>
            </div>
          )}
          {!ready && (
            <div className="image-tab__load">
              <ModelLoadPanel modelId={model.id} />
            </div>
          )}
          {settingsOpen && <SettingsPanel spec={spec} settings={settings} onChange={setSettings} />}
          <div className="composer__box">
            <textarea
              className="composer__input"
              placeholder={ready ? `Text for ${model.name} to speak` : 'Load the model to start'}
              rows={1}
              value={text}
              autoFocus
              onChange={(e) => setText(e.target.value)}
              onKeyDown={onKeyDown}
            />
            {hasSettings(spec) && (
              <button
                type="button"
                className={`button button--ghost button--small image-settings-toggle${settingsOpen ? ' image-settings-toggle--open' : ''}`}
                aria-expanded={settingsOpen}
                onClick={() => setSettingsOpen(!settingsOpen)}
              >
                Settings
              </button>
            )}
            {busy ? (
              <button className="composer__send composer__send--stop" type="button" onClick={() => running.get(session.id)?.()} aria-label="Stop">
                <StopIcon />
              </button>
            ) : (
              <button className="composer__send" type="submit" disabled={!text.trim() || !ready} aria-label={started ? 'Speak again' : 'Speak'}>
                <SendIcon />
              </button>
            )}
          </div>
          <p className="composer__hint">
            {summarise(spec, settings)} · Enter to {started ? 'speak again (replaces this clip)' : 'speak'}
          </p>
          {!started && (
            <div className="image-examples">
              {EXAMPLES.map((example) => (
                <button key={example} type="button" className="button button--ghost button--small" onClick={() => setText(example)}>
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

function ClipView({ spec, clip, sessionId }: { spec: SpeechModelSpec; clip: Clip; sessionId: string }) {
  const url = useClipUrl(sessionId, clip);
  const elapsed = useElapsedSeconds(clip.status === 'generating' ? clip.createdAt : null);
  const progress = clip.progress ? clip.progress.chunk / Math.max(1, clip.progress.chunks) : 0;
  const audio = useRef<HTMLAudioElement>(null);
  // A clip just made plays once by itself; one reopened later waits for its play button.
  useEffect(() => {
    if (!url || !audio.current || !autoplay.delete(clip.id)) return;
    void audio.current.play().catch(() => {});
  }, [url, clip.id]);
  const parts = clip.progress && clip.progress.chunks > 1 ? ` · part ${Math.min(clip.progress.chunk + 1, clip.progress.chunks)} of ${clip.progress.chunks}` : '';
  return (
    <figure className={`speech-view speech-view--${clip.status}`} key={clip.id}>
      <blockquote className="speech-view__text">{clip.text}</blockquote>
      {clip.status === 'generating' && (
        <div className="speech-view__progress">
          <span>
            Making the audio{parts} · {(clip.progress?.seconds ?? 0).toFixed(1)} s so far · {elapsed} s elapsed
          </span>
          <span className="image-view__bar">
            <span style={{ width: `${Math.max(0.05, progress) * 100}%` }} />
          </span>
        </div>
      )}
      {clip.status === 'done' && url && (
        <div className="speech-view__player">
          <audio controls src={url} ref={audio} />
          <a className="icon-button" href={url} download={`speech-${clip.id.slice(0, 8)}.wav`} title="Download" aria-label="Download audio">
            <DownloadIcon width={15} height={15} />
          </a>
        </div>
      )}
      <figcaption className="muted">
        {clip.status === 'failed' && clip.error ? <span className="image-view__error">{errorText(clip.error.code as never).title}. </span> : null}
        {clip.status === 'cancelled' ? 'Stopped before it finished. ' : ''}
        {summarise(spec, clip.settings)}
        {clip.seconds !== undefined ? ` · ${clip.seconds.toFixed(1)} s of audio` : ''}
        {clip.timings ? ` · made in ${(clip.timings.totalMs / 1000).toFixed(1)} s` : ''}
      </figcaption>
    </figure>
  );
}

/**
 * Starts speaking with the loaded model: the whole clip is made first (progress shown as it goes), then saved and
 * played. Streaming playback stuttered with the models slower than real time.
 */
function startSpeech(sessionId: string, spec: SpeechModelSpec, text: string, settings: SpeechSettings): void {
  running.get(sessionId)?.();
  const clip: Clip = {
    id: crypto.randomUUID(),
    text,
    settings: { ...settings },
    createdAt: Date.now(),
    status: 'generating',
    progress: { chunk: 0, chunks: 1, seconds: 0 },
  };
  speechStore.set(sessionId, clip);
  sessionStore.touch(sessionId);
  const current = () => {
    const c = speechStore.get(sessionId);
    return c?.id === clip.id ? c : null;
  };
  const pieces: Float32Array[] = [];
  let samples = 0;
  const handle = getRuntimeClient().generateSpeech({ text, settings }, (p) => {
    pieces.push(p.audio);
    samples += p.audio.length;
    const c = current();
    if (c) speechStore.set(sessionId, { ...c, progress: { chunk: p.chunk, chunks: p.chunks, seconds: samples / spec.sampleRate } });
  });
  running.set(sessionId, () => handle.cancel());
  handle.result
    .then(async (r) => {
      await speechStore.saveAudio(sessionId, clip.id, encodeWav(pieces, r.sampleRate));
      const c = current();
      if (!c) return;
      autoplay.add(clip.id);
      speechStore.set(sessionId, { ...c, status: 'done', progress: undefined, seconds: r.samples / r.sampleRate, timings: r.timings });
      sessionStore.touch(sessionId);
    })
    .catch((err: unknown) => {
      const c = current();
      if (!c) return;
      const code = err instanceof RuntimeError ? err.code : 'generation-failed';
      const message = err instanceof Error ? err.message : String(err);
      speechStore.set(sessionId, {
        ...c,
        status: code === 'cancelled' ? 'cancelled' : 'failed',
        progress: undefined,
        error: code === 'cancelled' ? undefined : { code, message },
      });
    })
    .finally(() => running.delete(sessionId));
}

function SettingsPanel({ spec, settings, onChange }: { spec: SpeechModelSpec; settings: SpeechSettings; onChange: (s: SpeechSettings) => void }) {
  return (
    <div className="image-settings">
      {spec.voices.length > 1 && (
        <label className="image-setting image-setting--select">
          <span className="image-setting__label">Voice</span>
          <select value={settings.voice} onChange={(e) => onChange({ ...settings, voice: e.target.value })}>
            {spec.voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </label>
      )}
      {spec.defaults.instruct !== undefined && (
        <label className="image-setting image-setting--text" title="Describe the voice: gender, age, accent, tone, pace.">
          <span className="image-setting__label">Voice description</span>
          <input
            type="text"
            value={settings.instruct ?? spec.defaults.instruct}
            maxLength={300}
            onChange={(e) => onChange({ ...settings, instruct: e.target.value })}
          />
        </label>
      )}
      {spec.languages && (
        <label className="image-setting image-setting--select" title="The language of the text.">
          <span className="image-setting__label">Language</span>
          <select value={settings.language ?? spec.defaults.language} onChange={(e) => onChange({ ...settings, language: e.target.value })}>
            {spec.languages.map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
              </option>
            ))}
          </select>
        </label>
      )}
      {spec.defaults.speed !== undefined && (
        <label className="image-setting image-setting--range" title="How fast the voice speaks.">
          <span className="image-setting__label">
            Speed<span className="image-setting__value">{(settings.speed ?? spec.defaults.speed).toFixed(2)}×</span>
          </span>
          <input
            type="range"
            min={0.7}
            max={1.6}
            step={0.05}
            value={settings.speed ?? spec.defaults.speed}
            onChange={(e) => onChange({ ...settings, speed: Number(e.target.value) })}
          />
        </label>
      )}
      {spec.defaults.steps !== undefined && (
        <label className="image-setting image-setting--range" title="More steps: slightly cleaner audio, made more slowly.">
          <span className="image-setting__label">
            Steps<span className="image-setting__value">{settings.steps ?? spec.defaults.steps}</span>
          </span>
          <input type="range" min={2} max={16} step={1} value={settings.steps ?? spec.defaults.steps} onChange={(e) => onChange({ ...settings, steps: Number(e.target.value) })} />
        </label>
      )}
      <button type="button" className="button button--ghost button--small image-settings__reset" onClick={() => onChange({ ...spec.defaults })}>
        Reset
      </button>
    </div>
  );
}

/** Whether the model takes any setting (a choice of voice or language, speed or steps). */
function hasSettings(spec: SpeechModelSpec): boolean {
  return (
    spec.voices.length > 1 || !!spec.languages || spec.defaults.instruct !== undefined || spec.defaults.speed !== undefined || spec.defaults.steps !== undefined
  );
}

/** One line describing the settings, e.g. "Female 1 · 1.05× · 8 steps". */
function summarise(spec: SpeechModelSpec, settings: SpeechSettings): string {
  const parts = [settings.instruct?.trim() ? 'Described voice' : (spec.voices.find((v) => v.id === settings.voice)?.label ?? settings.voice)];
  if (settings.language) parts.push(spec.languages?.find((l) => l.id === settings.language)?.label ?? settings.language);
  if (settings.speed !== undefined) parts.push(`${settings.speed.toFixed(2)}×`);
  if (settings.steps !== undefined) parts.push(`${settings.steps} steps`);
  return parts.join(' · ');
}

/** Stops any run in the session and drops its clip (before the session is deleted). */
export function forgetSpeechSession(sessionId: string): void {
  running.get(sessionId)?.();
  speechStore.forget(sessionId);
}
