import { Component, type ReactNode } from 'react';
import { forgetOpenView } from '../App';

/**
 * Catches an error while showing the app, so the page says so instead of going blank. Reloading starts from the home
 * screen: the app reopens the last chat, which may be what failed to show.
 */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown) {
    console.error('The app failed to show:', error);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <main className="empty app-error" role="alert">
        <h1 className="empty__title">Something went wrong</h1>
        <p className="empty__text muted">Quick Inference hit an error while showing this page. Your chats and models are still saved.</p>
        <div>
          <button
            className="button button--primary"
            type="button"
            onClick={() => {
              forgetOpenView();
              location.assign('/');
            }}
          >
            Reload
          </button>
        </div>
        <details className="error-details">
          <summary>Technical details</summary>
          <code>{error.message || error.name}</code>
        </details>
      </main>
    );
  }
}
