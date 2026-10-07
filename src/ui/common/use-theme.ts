import { useCallback, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

// Keep in sync with the inline script in index.html, which applies the theme before first paint.
const STORAGE_KEY = 'quick-inf:theme';
const darkQuery = () => window.matchMedia('(prefers-color-scheme: dark)');

function readStored(): Theme | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value === 'light' || value === 'dark' ? value : null;
  } catch {
    return null;
  }
}

/** Follows the OS setting until the user picks a theme, then remembers that choice. */
export function useTheme() {
  const [chosen, setChosen] = useState<Theme | null>(readStored);
  const [system, setSystem] = useState<Theme>(() => (darkQuery().matches ? 'dark' : 'light'));
  const theme = chosen ?? system;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    const query = darkQuery();
    const onChange = () => setSystem(query.matches ? 'dark' : 'light');
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  const toggleTheme = useCallback(() => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark';
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Storage unavailable: the choice still applies for this visit.
    }
    setChosen(next);
  }, [theme]);

  return { theme, toggleTheme };
}
