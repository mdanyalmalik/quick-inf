import type { RuntimeErrorCode } from '../../runtime/contract';

export interface ErrorText {
  title: string;
  /** What the user can do about it. */
  hint: string;
}

const TEXT: Record<RuntimeErrorCode, ErrorText> = {
  'too-big-for-cpu': {
    title: 'Too big to run here without a GPU',
    hint:
      'This browser has no GPU the app can use, and the model is too big for the CPU. Pick a smaller model, or use a recent Chrome or Edge with hardware acceleration turned on. If models ran on the GPU here a moment ago, the GPU was reset: fully close the browser (on a phone, swipe it away) and open it again.',
  },
  'incompatible-device': {
    title: 'This GPU can’t run the model',
    hint: 'The GPU is missing a feature this model needs.',
  },
  'download-failed': {
    title: 'The download failed',
    hint: 'Check your internet connection and try again.',
  },
  'integrity-mismatch': {
    title: 'The downloaded files didn’t match',
    hint: 'The download was corrupted or incomplete. Trying again usually fixes it.',
  },
  'load-failed': {
    title: 'The model couldn’t be loaded onto the GPU',
    hint: 'Closing other GPU-heavy tabs or apps may free enough memory. Then try again.',
  },
  'validation-failed': {
    title: 'The model failed its test run',
    hint: 'Try again. If it keeps failing, this GPU or browser may not be supported yet.',
  },
  'context-overflow': {
    title: 'That input is too long',
    hint: 'Shorten it or split it into smaller parts.',
  },
  'generation-failed': {
    title: 'Something went wrong while the model was running',
    hint: 'Try again.',
  },
  busy: {
    title: 'The model is busy',
    hint: 'Wait for the model to finish, or stop it.',
  },
  'not-loaded': {
    title: 'The model isn’t loaded',
    hint: 'Load the model, then try again.',
  },
  cancelled: {
    title: 'Cancelled',
    hint: '',
  },
  'device-lost': {
    title: 'The GPU stopped responding',
    hint: 'This can happen after sleep, a graphics driver reset, or running out of GPU memory. Loading again starts a fresh runtime. If that fails too (common on phones), fully close the browser and open it again.',
  },
  'out-of-memory': {
    title: 'Not enough GPU memory for this model',
    hint: 'Choose a smaller model, or close other tabs and apps that use the GPU, then try again.',
  },
  'runtime-crashed': {
    title: 'The inference runtime crashed',
    hint: 'Loading again starts a fresh runtime.',
  },
  'storage-full': {
    title: 'Not enough storage space',
    hint: 'Free up disk space, or delete other saved models, then try again.',
  },
  'storage-unavailable': {
    title: 'Models can’t be saved in this browser window',
    hint: 'Private or guest windows often block storage. Use a normal window to keep models on this device.',
  },
  'in-use-elsewhere': {
    title: 'Another tab is using this model',
    hint: 'Wait for the other Quick Inference tab to finish, or close it, then try again.',
  },
  'runtime-files-unavailable': {
    title: 'The app’s runtime files couldn’t be loaded',
    hint: 'Check your connection, or that the app’s server is running, then try again. Reloading the page also helps.',
  },
};

export function errorText(code: RuntimeErrorCode): ErrorText {
  return TEXT[code];
}
