import { Analytics } from '@vercel/analytics/react';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { getRuntimeClient } from './runtime/client';
import { currentBudget, fittedContext, memoryEvidence, trackMemoryEvidence } from './recommend/memory-evidence';
import { App } from './ui/App';
import { JoinScreen } from './ui/share/JoinScreen';
import { sharing } from './ui/share/SharePanel';
import './ui/styles.css';

trackMemoryEvidence(getRuntimeClient());

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* Another device's join link for a shared model (src/distributed). */}
    {location.pathname === '/join' ? <JoinScreen /> : <App />}
    {/* Vercel Web Analytics: page views only, no cookies. */}
    <Analytics />
  </StrictMode>,
);

if (import.meta.env.DEV) {
  // Lets scripted tests (and the console) simulate runtime failures.
  const client = getRuntimeClient();
  Object.assign(window, { __quickInf: { fault: client.debugFault.bind(client), client, evidence: memoryEvidence, currentBudget, fittedContext, sharing } });
}
