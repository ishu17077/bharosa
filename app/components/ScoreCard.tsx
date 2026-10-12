'use client';

import { useMemo } from 'react';

interface ScoreCardProps {
  title: string;
  score: number;
  maxScore?: number;
  icon: string;
  details: { label: string; value: string }[];
  color: string;
  signal: 'acoustic' | 'timing' | 'context';
}

export default function ScoreCard({
  title,
  score,
  maxScore = 1,
  icon,
  details,
  color,
}: ScoreCardProps) {
  const pct = useMemo(() => Math.round((score / maxScore) * 100), [score, maxScore]);
  const clampedPct = Math.min(100, Math.max(0, pct));

  // SVG arc parameters
  const size = 100;
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const arcLength = (clampedPct / 100) * circumference;
  const dashOffset = circumference - arcLength;

  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-white/5 bg-[#0d1117] p-5 min-w-0">
      {/* Header */}
      <div className="flex items-center gap-2 w-full">
        <span className="text-lg" role="img" aria-label={title}>
          {icon}
        </span>
        <span
          className="text-xs tracking-widest text-gray-400 uppercase"
          style={{ fontFamily: 'ui-monospace, monospace' }}
        >
          {title}
        </span>
      </div>

      {/* Circular gauge */}
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="transform -rotate-90"
        >
          {/* Background track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={strokeWidth}
          />
          {/* Score arc */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{ transition: 'stroke-dashoffset 0.6s ease-out, stroke 0.4s ease' }}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-xl font-bold tabular-nums"
            style={{
              fontFamily: 'ui-monospace, monospace',
              color,
              transition: 'color 0.4s ease',
            }}
          >
            {clampedPct}%
          </span>
        </div>
      </div>

      {/* Detail rows */}
      <div className="w-full flex flex-col gap-1.5 mt-1">
        {details.map((d) => (
          <div
            key={d.label}
            className="flex items-center justify-between text-xs"
          >
            <span className="text-gray-500 truncate">{d.label}</span>
            <span
              className="text-gray-300 tabular-nums font-medium"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              {d.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
