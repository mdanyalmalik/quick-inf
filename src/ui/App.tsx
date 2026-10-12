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
import { RecommendationsPage } from './recommend/RecommendationsPage';


/** Which pane fills the screen on phones. */
type Pane = 'models' | 'sessions' | 'main';

/** Everything that decides what the window shows. */
interface ViewState {
  modelId: string | null;
  sessionId: string | null;
  sessionPanelOpen: boolean;
  recommendationsOpen: boolean;
  pane: Pane;
}

/**
 * The view is saved per tab (sessionStorage), so reloading a tab shows exactly what it showed, whatever other
 * tabs do. A new tab or visit starts from the last model and chat opened in any tab (localStorage).
 */
const VIEW_KEY = 'quick-inf:view';
const LAST_OPEN_KEY = 'quick-inf:last-open';
const PANES: readonly Pane[] = ['models', 'sessions', 'main'];

function readView(): ViewState {
  const str = (v: unknown) => (typeof v === 'string' ? v : null);
  try {
    const tab = JSON.parse(sessionStorage.getItem(VIEW_KEY) ?? 'null') as Partial<Record<keyof ViewState, unknown>> | null;
    if (tab) {
      return {
        modelId: str(tab.modelId),
        sessionId: str(tab.sessionId),
        sessionPanelOpen: tab.sessionPanelOpen === true,
        recommendationsOpen: tab.recommendationsOpen === true,
        pane: PANES.includes(tab.pane as Pane) ? (tab.pane as Pane) : 'models',
      };
    }
  } catch {
    // Fall back to the last visit.
  }
  try {
    const last = JSON.parse(localStorage.getItem(LAST_OPEN_KEY) ?? 'null') as { modelId?: unknown; sessionId?: unknown } | null;
    const modelId = str(last?.modelId);
    const sessionId = str(last?.sessionId);
    return { modelId, sessionId, sessionPanelOpen: modelId !== null, recommendationsOpen: false, pane: sessionId ? 'main' : modelId ? 'sessions' : 'models' };
  } catch {
    return { modelId: null, sessionId: null, sessionPanelOpen: false, recommendationsOpen: false, pane: 'models' };
  }
}

export function App() {
  const models = useModelEntries();
  const sessions = useSyncExternalStore(sessionStore.subscribe, sessionStore.getAll);
  const [initial] = useState(readView);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(initial.modelId);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(initial.sessionId);
  const [sessionPanelOpen, setSessionPanelOpen] = useState(initial.sessionPanelOpen);
  const narrow = useMediaQuery(NARROW_QUERY);
  const [pane, setPane] = useState<Pane>(initial.pane);
  /** The device test results and recommendations replace the main area while open. */
  const [recommendationsOpen, setRecommendationsOpen] = useState(initial.recommendationsOpen);

  useEffect(() => {
    void sessionStore.init();
  }, []);

  useEffect(() => {
    const view: ViewState = { modelId: selectedModelId, sessionId: selectedSessionId, sessionPanelOpen, recommendationsOpen, pane };
    try {
      sessionStorage.setItem(VIEW_KEY, JSON.stringify(view));
      localStorage.setItem(LAST_OPEN_KEY, JSON.stringify({ modelId: selectedModelId, sessionId: selectedSessionId }));
    } catch {
      // Not remembering the view is fine.
    }
  }, [selectedModelId, selectedSessionId, sessionPanelOpen, recommendationsOpen, pane]);

  const { theme, toggleTheme } = useTheme();
  const firstRun = useFirstRun();
  const [welcomeOpen, setWelcomeOpen] = useState(firstRun.isFirstRun);

  const model = models.find((m) => m.id === selectedModelId) ?? null;
  const workspace = model ? workspaces[model.task] : null;
  const modelSessions = model ? sessions.filter((s) => s.modelId === model.id) : [];
  const session = modelSessions.find((s) => s.id === selectedSessionId) ?? null;

  function openRecommendations() {
    setRecommendationsOpen(true);
    setPane('main');
  }

  function closeRecommendations() {
    setRecommendationsOpen(false);
    setPane('models');
  }

  /** From a recommendation card: show that model's overview page. */
  function openModelFromRecommendations(id: string) {
    setRecommendationsOpen(false);
    setSelectedModelId(id);
    setSelectedSessionId(null);
    setSessionPanelOpen(true);
    setPane('main');
  }

  function selectModel(id: string) {
    setRecommendationsOpen(false);
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
    setRecommendationsOpen(false);
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
    setRecommendationsOpen(false);
    setSelectedModelId(null);
    setSelectedSessionId(null);
    setSessionPanelOpen(false);
    // Phones show the start page as its own screen (desktop always shows it beside the models).
    setPane(narrow ? 'main' : 'models');
  }

  /** The selected model's overview page (the page shown when a model is first picked). */
  function openModelOverview() {
    setRecommendationsOpen(false);
    setSelectedSessionId(null);
    setPane('main');
  }

  function selectSession(id: string) {
    setRecommendationsOpen(false);
    setSelectedSessionId(id);
    setPane('main');
  }

  function renameSession(id: string, title: string) {
    sessionStore.rename(id, title);
  }

  function closeWelcome() {
    firstRun.markComplete();
    setWelcomeOpen(false);
  }

  function welcomeTestDone() {
    closeWelcome();
    openRecommendations();
  }

  // Phones show one pane at a time; fall back to an earlier pane if its content is gone.
  const visible: Pane = narrow
    ? (recommendationsOpen || !model) && pane === 'main'
      ? 'main'
      : pane !== 'models' && model
        ? pane
        : 'models'
    : 'main';
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
          onOpenBenchmark={openRecommendations}
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
          onRenameSession={renameSession}
          onOpenModel={openModelOverview}
          onClose={() => (narrow ? setPane('models') : setSessionPanelOpen(false))}
          narrow={narrow}
        />
      )}

      {showMain && (
        <main className="main">
          {recommendationsOpen ? (
            <RecommendationsPage
              initialTab={model?.task}
              onOpenModel={openModelFromRecommendations}
              onBack={narrow ? closeRecommendations : undefined}
            />
          ) : !model || !workspace ? (
            <NoModelSelected onBack={narrow ? () => setPane('models') : undefined} />
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

      {welcomeOpen && <BenchmarkPrompt onClose={closeWelcome} onComplete={welcomeTestDone} />}
    </div>
  );
}
