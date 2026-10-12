import { useRef } from 'react';
import { DownloadIcon, UploadIcon } from '../common/icons';
import { useChatBackup } from './use-chat-backup';

/** Home page: back chats up to a file, or restore them from one. */
export function ChatBackupPanel() {
  const backup = useChatBackup();
  const fileInput = useRef<HTMLInputElement>(null);
  const { backedUpAt } = backup.status;

  return (
    <section className="home-storage chat-backup" aria-labelledby="chat-backup-title">
      <h2 id="chat-backup-title" className="home-storage__title">
        Your work
      </h2>
      <p>
        {backup.chatCount === 0
          ? 'Nothing saved in this browser yet.'
          : `${backup.savedText} saved in this browser.`}{' '}
        {backup.needsBackup ? (
          <span className="chat-backup__unsaved">
            {backedUpAt === null ? 'Not backed up yet.' : 'Changed since the last backup.'}
          </span>
        ) : backedUpAt !== null ? (
          <span className="muted">Last backup {formatWhen(backedUpAt)}.</span>
        ) : null}
      </p>
      <p className="home-storage__note muted">
        Browsers can clear saved data, so save a backup file to keep your work. Models can always be downloaded again.
      </p>
      <div className="chat-backup__actions">
        <button
          className="button button--small"
          type="button"
          onClick={() => void backup.save()}
          disabled={backup.busy || backup.chatCount === 0}
        >
          <DownloadIcon width={15} height={15} />
          Save backup
        </button>
        <button className="button button--small" type="button" onClick={() => fileInput.current?.click()} disabled={backup.busy}>
          <UploadIcon width={15} height={15} />
          Restore backup
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".json,application/json"
          hidden
          aria-label="Backup file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = ''; // the same file can be picked again
            if (file) void backup.restore(file);
          }}
        />
      </div>
      {backup.message && (
        <p className={`home-storage__note ${backup.message.kind === 'error' ? 'chat-backup__error' : 'muted'}`} role="status">
          {backup.message.text}
        </p>
      )}
    </section>
  );
}

function formatWhen(at: number): string {
  const date = new Date(at);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay
    ? `today at ${date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}`
    : `on ${date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}`;
}
