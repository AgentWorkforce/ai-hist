const HYDRATABLE_SOURCES = new Set(['claude', 'codex', 'cursor', 'grok', 'opencode']);

/** Select a recent, provider-diverse set, preferring sessions not yet hydrated. */
export function selectHydrationSessions(sessions, count) {
  const eligible = sessions.filter((session) => (
    HYDRATABLE_SOURCES.has(session.source)
    && session.locations?.includes('local')
    && typeof session.rawPath === 'string'
    && session.rawPath.length > 0
  ));
  const selected = [];
  for (const discoveryState of ['shallow', 'full']) {
    const groups = new Map();
    for (const session of eligible) {
      if (session.discoveryState !== discoveryState) continue;
      const group = groups.get(session.source) ?? [];
      group.push(session);
      groups.set(session.source, group);
    }
    while (selected.length < count && [...groups.values()].some((group) => group.length > 0)) {
      for (const group of groups.values()) {
        const session = group.shift();
        if (session) selected.push(session);
        if (selected.length === count) break;
      }
    }
    if (selected.length === count) break;
  }
  return selected;
}

/** Summarize measured milliseconds using nearest-rank percentiles. */
export function durationSummary(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const percentile = (fraction) => sorted[Math.ceil(sorted.length * fraction) - 1];
  return {
    samples: sorted.length,
    minMs: Number(sorted[0].toFixed(2)),
    meanMs: Number((sorted.reduce((sum, value) => sum + value, 0) / sorted.length).toFixed(2)),
    p50Ms: Number(percentile(0.5).toFixed(2)),
    p95Ms: Number(percentile(0.95).toFixed(2)),
    maxMs: Number(sorted.at(-1).toFixed(2)),
  };
}
