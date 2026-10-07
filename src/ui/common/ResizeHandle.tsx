import type { KeyboardEvent, PointerEvent } from 'react';
import type { WidthBounds } from './use-resizable-width';

interface ResizeHandleProps {
  label: string;
  width: number;
  bounds: WidthBounds;
  onResize: (width: number) => void;
}

const KEYBOARD_STEP = 16;

/** Drag handle sitting on a panel's right edge. Double-click restores the default width. */
export function ResizeHandle({ label, width, bounds, onResize }: ResizeHandleProps) {
  function onPointerDown(e: PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return;
    e.preventDefault();
    const handle = e.currentTarget;
    const startX = e.clientX;
    const startWidth = width;
    handle.setPointerCapture(e.pointerId);
    document.documentElement.classList.add('is-resizing');

    const onMove = (ev: globalThis.PointerEvent) => onResize(startWidth + ev.clientX - startX);
    const onUp = () => {
      document.documentElement.classList.remove('is-resizing');
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', onUp);
      handle.removeEventListener('pointercancel', onUp);
    };
    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
  }

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key === 'ArrowLeft') onResize(width - KEYBOARD_STEP);
    else if (e.key === 'ArrowRight') onResize(width + KEYBOARD_STEP);
    else return;
    e.preventDefault();
  }

  return (
    <div
      className="resize-handle"
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={width}
      aria-valuemin={bounds.min}
      aria-valuemax={bounds.max}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
      onDoubleClick={() => onResize(bounds.initial)}
    />
  );
}
