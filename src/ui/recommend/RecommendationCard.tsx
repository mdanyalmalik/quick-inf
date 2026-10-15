import { ARENA_SOURCE, artifactBytes, engineName, INTELLIGENCE_SOURCE } from '../../catalogue/models';
import type { Recommendation, Verdict } from '../../recommend/recommend';
import { formatBytes, formatMs } from '../common/format';

const VERDICT_LABEL: Record<Verdict, string> = {
  good: 'Likely runs well',
  slow: 'May be slow',
  'may-not-fit': 'May not fit',
  failed: 'Didn’t load here',
  unsupported: 'Not supported on this GPU',
  untested: 'Not tested yet',
};

const VERDICT_SHORT: Record<Verdict, string> = {
  good: 'Runs well',
  slow: 'May be slow',
  'may-not-fit': 'May not fit',
  failed: 'Didn’t load',
  unsupported: 'Not supported',
  untested: '',
};

export function VerdictBadge({ verdict, short = false }: { verdict: Verdict; short?: boolean }) {
  const label = short ? VERDICT_SHORT[verdict] : VERDICT_LABEL[verdict];
  if (!label) return null;
  return <span className={`verdict verdict--${verdict}`}>{label}</span>;
}

const TOKEN_HINT = 'A token is a word or part of a word; roughly ¾ of a word on average.';

interface RecommendationCardProps {
  rec: Recommendation;
  /** Show the model's name and an Open button (the recommendations page). */
  onOpen?: () => void;
}

/** What to expect from one model on this device: verdict, speed, memory and published quality. */
export function RecommendationCard({ rec, onOpen }: RecommendationCardProps) {
  const { model, estimate, measured } = rec;
  const tps = measured.tokensPerSecond ?? estimate?.tokensPerSecond;
  const shortMs = measured.firstTokenShortMs ?? estimate?.firstTokenShortMs;
  const longMs = measured.firstTokenLongMs ?? estimate?.firstTokenLongMs;
  const download = rec.artifact ? artifactBytes(model, rec.artifact) : undefined;

  return (
    <article className={`rec-card${rec.recommended ? ' rec-card--recommended' : ''}`}>
      <header className="rec-card__header">
        {onOpen && <h2 className="rec-card__name">{model.name}</h2>}
        <VerdictBadge verdict={rec.verdict} />
        {rec.recommended && <span className="badge">Recommended</span>}
        {onOpen && (
          <button className="button button--ghost button--small rec-card__open" type="button" onClick={onOpen}>
            Open
          </button>
        )}
      </header>
      {onOpen && (
        <p className="rec-card__meta muted">
          {model.publisher} · {model.parameters} parameters
          {rec.artifact ? ` · ${engineName(rec.artifact.engine)}, ${rec.artifact.precision}` : ''}
        </p>
      )}

      {rec.reasons.length > 0 && (
        <ul className="rec-card__reasons">
          {rec.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}

      {(estimate || measured.tokensPerSecond) && (
        <dl className="rec-card__stats">
          <Stat
            label="Generation speed"
            value={tps ? `${Math.round(tps)} tokens/s` : '—'}
            source={measured.tokensPerSecond ? 'measured' : 'estimated'}
            title={TOKEN_HINT}
          />
          <Stat
            label="First word, short prompt"
            value={shortMs ? formatMs(shortMs) : '—'}
            source={measured.firstTokenShortMs ? 'measured' : 'estimated'}
          />
          <Stat
            label="First word, long prompt"
            value={longMs ? formatMs(longMs) : '—'}
            source={measured.firstTokenLongMs ? 'measured' : 'estimated'}
            title="A prompt of about 2,000 tokens (roughly 1,500 words)."
          />
          <Stat
            label="Memory needed"
            value={rec.memoryBytes ? formatBytes(rec.memoryBytes) : '—'}
            source="estimated"
            title="Model weights plus the conversation's working memory at its default length."
          />
          {download !== undefined && <Stat label="Download" value={formatBytes(download)} />}
        </dl>
      )}

      <div className="rec-card__quality">
        <p>{model.quality.summary}</p>
        {(model.quality.arena !== undefined || model.quality.intelligence !== undefined) && (
          <p className="rec-card__scores muted">
            Independent ratings:{' '}
            {model.quality.arena !== undefined && <span title={`Source: ${ARENA_SOURCE}`}>people’s votes (LMArena) {model.quality.arena}</span>}
            {model.quality.arena !== undefined && model.quality.intelligence !== undefined && ' · '}
            {model.quality.intelligence !== undefined && (
              <span title={`Source: ${INTELLIGENCE_SOURCE}`}>benchmarks (Artificial Analysis index) {model.quality.intelligence}</span>
            )}
            <span className="rec-card__source"> (hover for the source)</span>
          </p>
        )}
        <p className="rec-card__scores muted">
          Published scores:{' '}
          {model.quality.scores.map((s, i) => (
            <span key={s.benchmark} title={`Source: ${s.source}`}>
              {i > 0 && ' · '}
              {s.benchmark} {s.score}
            </span>
          ))}
          <span className="rec-card__source"> (from the model makers; hover for the source)</span>
        </p>
      </div>
    </article>
  );
}

function Stat({ label, value, source, title }: { label: string; value: string; source?: string; title?: string }) {
  return (
    <div title={title}>
      <dt>{label}</dt>
      <dd>
        {value}
        {source && <span className="rec-card__source"> {source}</span>}
      </dd>
    </div>
  );
}
