import { useEffect, useState, useSyncExternalStore } from 'react';
import { sessionStore } from '../conversations/session-store';
import { useModelEntries } from './models/model-entries';
import { workspaces } from './workspaces/registry';
import { ModelSidebar } from './models/ModelSidebar';
import { SessionPanel } from './sessions/SessionPanel';
import { BenchmarkPrompt } from './onboarding/BenchmarkPrompt';
import { useFirstRun } from './onboarding/use-first-run';
import { useTheme } from './common/use-theme';
import { NARROW_QUERY, useMediaQuery } from './common/use-media-query';
import { NoModelSelected, NoSessionSelected } from './EmptyStates';

type BenchmarkDialog = 'welcome' | 'rerun' | null;

/** The last open model and session, restored on the next visit (this browser only). */
const LAST_OPEN_KEY = 'quick-inf:last-open';

function readLastOpen(): { modelId: string | null; sessionId: string | null } {
  try {
    const value = JSON.parse(localStorage.getItem(LAST_OPEN_KEY) ?? 'null') as { modelId?: unknown; sessionId?: unknown } | null;
    return {
      modelId: typeof value?.modelId === 'string' ? value.modelId : null,
      sessionId: typeof value?.sessionId === 'string' ? value.sessionId : null,
    };
  } catch {
    return { modelId: null, sessionId: null };
  }
}
/** Which pane fills the screen on phones. */
type Pane = 'models' | 'sessions' | 'main';

export function App() {
  const models = useModelEntries();
  const sessions = useSyncExternalStore(sessionStore.subscribe, sessionStore.getAll);
  const [lastOpen] = useState(readLastOpen);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(lastOpen.modelId);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(lastOpen.sessionId);
  const [sessionPanelOpen, setSessionPanelOpen] = useState(lastOpen.modelId !== null);
  const narrow = useMediaQuery(NARROW_QUERY);
  const [pane, setPane] = useState<Pane>(lastOpen.sessionId ? 'main' : lastOpen.modelId ? 'sessions' : 'models');

  useEffect(() => {
    void sessionStore.init();
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(LAST_OPEN_KEY, JSON.stringify({ modelId: selectedModelId, sessionId: selectedSessionId }));
    } catch {
      // Not remembering the last chat is fine.
    }
  }, [selectedModelId, selectedSessionId]);

  const { theme, toggleTheme } = useTheme();
  const firstRun = useFirstRun();
  const [benchmarkDialog, setBenchmarkDialog] = useState<BenchmarkDialog>(firstRun.isFirstRun ? 'welcome' : null);

  const model = models.find((m) => m.id === selectedModelId) ?? null;
  const workspace = model ? workspaces[model.task] : null;
  const modelSessions = model ? sessions.filter((s) => s.modelId === model.id) : [];
  const session = modelSessions.find((s) => s.id === selectedSessionId) ?? null;

  function selectModel(id: string) {
    if (narrow) {
      if (id !== selectedModelId) setSelectedSessionId(null);
      setSelectedModelId(id);
      setSessionPanelOpen(true);
      setPane('sessions');
      return;
    }
    if (id === selectedModelId) {
      setSessionPanelOpen((open) => !open);
      return;
    }
    setSelectedModelId(id);
    setSelectedSessionId(null);
    setSessionPanelOpen(true);
  }

  function newSession() {
    if (!model || !workspace) return;
    const created = sessionStore.create(model.id, `New ${workspace.sessionLabel.singular}`);
    setSelectedSessionId(created.id);
    setPane('main');
  }

  function deleteSession(id: string) {
    workspace?.onDeleteSession?.(id);
    void sessionStore.delete(id);
    if (id === selectedSessionId) {
      setSelectedSessionId(null);
      setPane('sessions');
    }
  }

  /** The start page: no model selected. */
  function goHome() {
    setSelectedModelId(null);
    setSelectedSessionId(null);
    setSessionPanelOpen(false);
    setPane('models');
  }

  /** The selected model's overview page (the page shown when a model is first picked). */
  function openModelOverview() {
    setSelectedSessionId(null);
    setPane('main');
  }

  function selectSession(id: string) {
    setSelectedSessionId(id);
    setPane('main');
  }

  function renameSession(id: string, title: string) {
    sessionStore.rename(id, title);
  }

  function closeBenchmark() {
    if (benchmarkDialog === 'welcome') firstRun.markComplete();
    setBenchmarkDialog(null);
  }

  // Phones show one pane at a time; fall back to an earlier pane if its content is gone.
  const visible: Pane = narrow ? (pane !== 'models' && model ? pane : 'models') : 'main';
  const showModels = !narrow || visible === 'models';
  const showSessions = narrow ? visible === 'sessions' : sessionPanelOpen;
  const showMain = !narrow || visible === 'main';

  return (
    <div className={`app${narrow ? ' app--narrow' : ''}`}>
      {showModels && (
        <ModelSidebar
          models={models}
          selectedModelId={selectedModelId}
          onSelectModel={selectModel}
          onOpenBenchmark={() => setBenchmarkDialog('rerun')}
          theme={theme}
          onToggleTheme={toggleTheme}
          onHome={goHome}
        />
      )}

      {model && workspace && showSessions && (
        <SessionPanel
          model={model}
          workspace={workspace}
          sessions={modelSessions}
          selectedSessionId={selectedSessionId}
          onSelectSession={selectSession}
          onNewSession={newSession}
          onDeleteSession={deleteSession}
          onOpenModel={openModelOverview}
          onClose={() => (narrow ? setPane('models') : setSessionPanelOpen(false))}
          narrow={narrow}
        />
      )}

      {showMain && (
        <main className="main">
          {!model || !workspace ? (
            <NoModelSelected />
          ) : !session ? (
            <NoSessionSelected
              model={model}
              workspace={workspace}
              onNewSession={newSession}
              onBack={narrow ? () => setPane('sessions') : undefined}
            />
          ) : (
            <workspace.Workspace
              key={session.id}
              model={model}
              session={session}
              onRename={(title) => renameSession(session.id, title)}
              onBack={narrow ? () => setPane('sessions') : undefined}
            />
          )}
        </main>
      )}

      {benchmarkDialog && <BenchmarkPrompt variant={benchmarkDialog} onClose={closeBenchmark} />}
    </div>
  );
}
