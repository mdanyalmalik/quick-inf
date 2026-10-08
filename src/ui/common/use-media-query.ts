import { useSyncExternalStore } from 'react';

/** Whether a CSS media query currently matches; re-renders when it changes. */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const list = matchMedia(query);
      list.addEventListener('change', onChange);
      return () => list.removeEventListener('change', onChange);
    },
    () => matchMedia(query).matches,
  );
}

/** Phone-sized screens: one pane at a time (models → chats → chat). Keep in sync with styles.css. */
export const NARROW_QUERY = '(max-width: 720px)';
