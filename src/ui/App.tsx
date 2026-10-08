import { useState } from 'react';
import type { Session } from './types';
import { modelEntries } from './models/model-entries';
import { workspaces } from './workspaces/registry';
import { ModelSidebar } from './models/ModelSidebar';
import { SessionPanel } from './sessions/SessionPanel';
import { BenchmarkPrompt } from './onboarding/BenchmarkPrompt';
import { useFirstRun } from './onboarding/use-first-run';
import { useTheme } from './common/use-theme';
import { NoModelSelected, NoSessionSelected } from './EmptyStates';

type BenchmarkDialog = 'welcome' | 'rerun' | null;

export function App() {
  const models = modelEntries;
  // In memory until conversations are persisted (milestone 4).
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [sessionPanelOpen, setSessionPanelOpen] = useState(false);

  const { theme, toggleTheme } = useTheme();
  const firstRun = useFirstRun();
  const [benchmarkDialog, setBenchmarkDialog] = useState<BenchmarkDialog>(firstRun.isFirstRun ? 'welcome' : null);

  const model = models.find((m) => m.id === selectedModelId) ?? null;
  const workspace = model ? workspaces[model.task] : null;
  const modelSessions = model ? sessions.filter((s) => s.modelId === model.id) : [];
  const session = modelSessions.find((s) => s.id === selectedSessionId) ?? null;

  function selectModel(id: string) {
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
    const created: Session = {
      id: crypto.randomUUID(),
      modelId: model.id,
      title: `New ${workspace.sessionLabel.singular}`,
      updatedAt: Date.now(),
    };
    setSessions((prev) => [created, ...prev]);
    setSelectedSessionId(created.id);
  }

  function renameSession(id: string, title: string) {
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, title, updatedAt: Date.now() } : s)));
  }

  function closeBenchmark() {
    if (benchmarkDialog === 'welcome') firstRun.markComplete();
    setBenchmarkDialog(null);
  }

  return (
    <div className="app">
      <ModelSidebar
        models={models}
        selectedModelId={selectedModelId}
        onSelectModel={selectModel}
        onOpenBenchmark={() => setBenchmarkDialog('rerun')}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {model && workspace && sessionPanelOpen && (
        <SessionPanel
          model={model}
          workspace={workspace}
          sessions={modelSessions}
          selectedSessionId={selectedSessionId}
          onSelectSession={setSelectedSessionId}
          onNewSession={newSession}
          onClose={() => setSessionPanelOpen(false)}
        />
      )}

      <main className="main">
        {!model || !workspace ? (
          <NoModelSelected />
        ) : !session ? (
          <NoSessionSelected model={model} workspace={workspace} onNewSession={newSession} />
        ) : (
          <workspace.Workspace
            key={session.id}
            model={model}
            session={session}
            onRename={(title) => renameSession(session.id, title)}
          />
        )}
      </main>

      {benchmarkDialog && <BenchmarkPrompt variant={benchmarkDialog} onClose={closeBenchmark} />}
    </div>
  );
}
