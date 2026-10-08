import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { getRuntimeClient } from './runtime/client';
import { App } from './ui/App';
import './ui/styles.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

if (import.meta.env.DEV) {
  // Lets scripted tests (and the console) simulate runtime failures.
  const client = getRuntimeClient();
  Object.assign(window, { __quickInf: { fault: client.debugFault.bind(client), client } });
}
