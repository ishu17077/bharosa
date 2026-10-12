'use client';

import { useMemo } from 'react';
import type { ReasonTag } from '@/app/lib/types';

interface ReasonTagsProps {
  tags: ReasonTag[];
  isActive: boolean;
}

const SEVERITY_CONFIG: Record<
  ReasonTag['severity'],
  { bg: string; border: string; text: string; order: number }
> = {
  high: {
    bg: 'rgba(220, 38, 38, 0.15)',
    border: 'rgba(220, 38, 38, 0.4)',
    text: '#fca5a5',
    order: 0,
  },
  medium: {
    bg: 'rgba(234, 179, 8, 0.12)',
    border: 'rgba(234, 179, 8, 0.35)',
    text: '#fcd34d',
    order: 1,
  },
  low: {
    bg: 'rgba(59, 130, 246, 0.12)',
    border: 'rgba(59, 130, 246, 0.35)',
    text: '#93c5fd',
    order: 2,
  },
};

const SIGNAL_ICON: Record<ReasonTag['signal'], string> = {
  acoustic: '🔊',
  timing: '⏱️',
  context: '🏦',
};

export default function ReasonTags({ tags, isActive }: ReasonTagsProps) {
  const sorted = useMemo(
    () =>
      [...tags].sort(
        (a, b) => SEVERITY_CONFIG[a.severity].order - SEVERITY_CONFIG[b.severity].order,
      ),
    [tags],
  );

  const counts = useMemo(() => {
    const c = { high: 0, medium: 0, low: 0 };
    for (const t of tags) c[t.severity]++;
    return c;
  }, [tags]);

  return (
    <div className="rounded-xl border border-white/5 bg-[#0d1117] p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span
          className="text-xs tracking-widest text-gray-400 uppercase"
          style={{ fontFamily: 'ui-monospace, monospace' }}
        >
          Analysis Reason Tags
        </span>

        {tags.length > 0 && (
          <div className="flex items-center gap-2">
            {counts.high > 0 && (
              <span
                className="text-xs px-1.5 py-0.5 rounded"
                style={{
                  background: SEVERITY_CONFIG.high.bg,
                  color: SEVERITY_CONFIG.high.text,
                  fontFamily: 'ui-monospace, monospace',
                }}
              >
                {counts.high} high
              </span>
            )}
            {counts.medium > 0 && (
              <span
                className="text-xs px-1.5 py-0.5 rounded"
                style={{
                  background: SEVERITY_CONFIG.medium.bg,
                  color: SEVERITY_CONFIG.medium.text,
                  fontFamily: 'ui-monospace, monospace',
                }}
              >
                {counts.medium} med
              </span>
            )}
            {counts.low > 0 && (
              <span
                className="text-xs px-1.5 py-0.5 rounded"
                style={{
                  background: SEVERITY_CONFIG.low.bg,
                  color: SEVERITY_CONFIG.low.text,
                  fontFamily: 'ui-monospace, monospace',
                }}
              >
                {counts.low} low
              </span>
            )}
          </div>
        )}
      </div>

      {/* Tags or empty state */}
      {!isActive || sorted.length === 0 ? (
        <div className="flex items-center gap-2 py-3 justify-center text-sm text-green-400/70">
          <span>✅</span>
          <span>No anomalies detected</span>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {sorted.map((tag, i) => {
            const cfg = SEVERITY_CONFIG[tag.severity];
            return (
              <span
                key={`${tag.code}-${i}`}
                className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-full"
                style={{
                  background: cfg.bg,
                  border: `1px solid ${cfg.border}`,
                  color: cfg.text,
                  fontFamily: 'ui-monospace, monospace',
                  animation: `tag-in 0.3s ease-out ${i * 0.05}s both`,
                }}
              >
                <span role="img" aria-label={tag.signal}>
                  {SIGNAL_ICON[tag.signal]}
                </span>
                {tag.label}
              </span>
            );
          })}
        </div>
      )}

      {/* Keyframe for tag entrance animation */}
      <style jsx>{`
        @keyframes tag-in {
          from {
            opacity: 0;
            transform: translateY(6px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
