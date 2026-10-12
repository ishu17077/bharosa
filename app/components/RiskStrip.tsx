'use client';

import { useMemo } from 'react';
import type { RiskTier } from '@/app/lib/types';

interface RiskStripProps {
  tier: RiskTier;
  score: number;
  isActive: boolean;
  latencyMs: number;
}

const TIER_CONFIG: Record<
  RiskTier,
  { from: string; to: string; icon: string; label: string; textColor: string }
> = {
  GREEN: {
    from: '#0d4f1c',
    to: '#1a7a3a',
    icon: '🛡️',
    label: 'LOW RISK',
    textColor: '#6ee7b7',
  },
  AMBER: {
    from: '#7a5a00',
    to: '#c49000',
    icon: '⚠️',
    label: 'MEDIUM RISK',
    textColor: '#fcd34d',
  },
  RED: {
    from: '#7a0d0d',
    to: '#c42020',
    icon: '🚨',
    label: 'HIGH RISK',
    textColor: '#fca5a5',
  },
};

export default function RiskStrip({
  tier,
  score,
  isActive,
  latencyMs,
}: RiskStripProps) {
  const config = useMemo(() => TIER_CONFIG[tier], [tier]);
  const pct = Math.round(score * 100);

  if (!isActive) {
    return (
      <div
        className="w-full flex items-center justify-center gap-3 rounded-lg border border-white/5"
        style={{
          height: 64,
          background: 'linear-gradient(90deg, #1f2937, #374151)',
        }}
      >
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-gray-500" />
        <span
          className="text-sm tracking-widest text-gray-400"
          style={{ fontFamily: 'ui-monospace, monospace' }}
        >
          STANDBY
        </span>
      </div>
    );
  }

  return (
    <div
      className="w-full flex items-center justify-between px-5 rounded-lg border border-white/10 relative overflow-hidden"
      style={{
        height: 64,
        background: `linear-gradient(90deg, ${config.from}, ${config.to})`,
        transition: 'background 0.6s ease',
      }}
    >
      {/* Pulsing overlay for RED tier */}
      {tier === 'RED' && (
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'rgba(220, 38, 38, 0.15)',
            animation: 'risk-pulse 1.2s ease-in-out infinite',
          }}
        />
      )}

      {/* Left side: icon + tier label */}
      <div className="flex items-center gap-3 z-10">
        <span className="text-2xl" role="img" aria-label={tier}>
          {config.icon}
        </span>
        <div className="flex flex-col">
          <span
            className="text-xs tracking-widest opacity-70"
            style={{ fontFamily: 'ui-monospace, monospace', color: config.textColor }}
          >
            CONFIDENCE TIER
          </span>
          <span
            className="text-base font-bold tracking-wider"
            style={{ color: config.textColor }}
          >
            {config.label}
          </span>
        </div>
      </div>

      {/* Center: score */}
      <div className="flex items-center gap-6 z-10">
        <div className="flex flex-col items-center">
          <span
            className="text-xs tracking-widest opacity-60"
            style={{ fontFamily: 'ui-monospace, monospace', color: config.textColor }}
          >
            SCORE
          </span>
          <span
            className="text-2xl font-bold tabular-nums"
            style={{ fontFamily: 'ui-monospace, monospace', color: config.textColor }}
          >
            {pct}%
          </span>
        </div>

        <div className="flex flex-col items-center">
          <span
            className="text-xs tracking-widest opacity-60"
            style={{ fontFamily: 'ui-monospace, monospace', color: config.textColor }}
          >
            LATENCY
          </span>
          <span
            className="text-lg font-semibold tabular-nums"
            style={{ fontFamily: 'ui-monospace, monospace', color: config.textColor }}
          >
            {latencyMs}ms
          </span>
        </div>
      </div>

      {/* Right side: pulsing indicator */}
      <div className="flex items-center gap-2 z-10">
        <span
          className="inline-block w-3 h-3 rounded-full"
          style={{
            backgroundColor: config.textColor,
            animation: 'dot-pulse 1.5s ease-in-out infinite',
          }}
        />
        <span
          className="text-xs tracking-wider opacity-70"
          style={{ fontFamily: 'ui-monospace, monospace', color: config.textColor }}
        >
          LIVE
        </span>
      </div>

      {/* Keyframe styles */}
      <style jsx>{`
        @keyframes risk-pulse {
          0%,
          100% {
            opacity: 0;
          }
          50% {
            opacity: 1;
          }
        }
        @keyframes dot-pulse {
          0%,
          100% {
            opacity: 1;
            transform: scale(1);
          }
          50% {
            opacity: 0.4;
            transform: scale(0.7);
          }
        }
      `}</style>
    </div>
  );
}
