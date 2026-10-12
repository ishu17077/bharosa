'use client';

import { useState } from 'react';
import { CRMAction, RiskTier } from '@/app/lib/types';

export interface TestScenario {
  id: string;
  name: string;
  description: string;
  actionType: CRMAction;
  expectedTier: RiskTier;
  icon: string;
}

const SCENARIOS: TestScenario[] = [
  {
    id: 'legitimate',
    name: 'Legitimate Caller',
    description:
      'Genuine customer calling for a routine balance inquiry. Natural speech patterns with normal acoustic characteristics.',
    actionType: 'BALANCE_INQUIRY',
    expectedTier: 'GREEN',
    icon: '👤',
  },
  {
    id: 'clone-attack',
    name: 'Voice Clone Attack',
    description:
      'Synthetic voice attempting a high-risk SIM change. Exhibits flat spectral profile and unnaturally consistent pitch.',
    actionType: 'SIM_CHANGE',
    expectedTier: 'RED',
    icon: '🎭',
  },
  {
    id: 'noisy-legit',
    name: 'Noisy Legitimate Caller',
    description:
      'Real customer in a noisy environment requesting card block. Background noise may affect acoustic scores.',
    actionType: 'CARD_BLOCK',
    expectedTier: 'AMBER',
    icon: '🔊',
  },
];

type ScenarioStatus = 'ready' | 'running' | 'completed';

interface CallSimulatorProps {
  onScenarioSelect: (scenario: TestScenario) => void;
  isActive: boolean;
}

const TIER_COLORS: Record<RiskTier, string> = {
  GREEN: 'text-green-400 bg-green-500/10 border-green-500/30',
  AMBER: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
  RED: 'text-red-400 bg-red-500/10 border-red-500/30',
};

export default function CallSimulator({
  onScenarioSelect,
  isActive,
}: CallSimulatorProps) {
  const [statuses, setStatuses] = useState<Record<string, ScenarioStatus>>(() =>
    Object.fromEntries(SCENARIOS.map((s) => [s.id, 'ready' as ScenarioStatus]))
  );
  const [activeId, setActiveId] = useState<string | null>(null);

  const handleSelect = (scenario: TestScenario) => {
    // Mark previous as completed, new one as running
    setStatuses((prev) => {
      const next = { ...prev };
      if (activeId && activeId !== scenario.id) {
        next[activeId] = 'completed';
      }
      next[scenario.id] = 'running';
      return next;
    });
    setActiveId(scenario.id);
    onScenarioSelect(scenario);
  };

  return (
    <div
      className={`rounded-xl border p-4 transition-opacity duration-300 ${
        isActive
          ? 'border-[var(--card-border)] bg-[var(--card-bg)]'
          : 'border-[var(--card-border)]/50 bg-[var(--card-bg)]/50 opacity-60'
      }`}
    >
      {/* Header */}
      <div className="mb-3">
        <h3 className="text-xs font-semibold tracking-widest text-[var(--purple)] uppercase">
          Test Scenarios
        </h3>
        <p className="mt-1 text-[11px] text-slate-400">
          Select a scenario to configure action context and expected outcome
        </p>
      </div>

      {/* Scenario cards */}
      <div className="flex flex-col gap-2">
        {SCENARIOS.map((scenario) => {
          const status = statuses[scenario.id];
          const isRunning = status === 'running';
          const tierClass = TIER_COLORS[scenario.expectedTier];

          return (
            <button
              key={scenario.id}
              onClick={() => handleSelect(scenario)}
              className={`group relative flex items-start gap-3 rounded-lg border px-3 py-2.5 text-left transition-all duration-200 ${
                isRunning
                  ? 'border-[var(--cyan)]/50 bg-[var(--cyan)]/5 ring-1 ring-[var(--cyan)]/20'
                  : 'border-[var(--card-border)] bg-[var(--background)] hover:border-slate-500'
              }`}
            >
              {/* Icon */}
              <span className="mt-0.5 text-xl leading-none">{scenario.icon}</span>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-white">
                    {scenario.name}
                  </span>
                  {/* Status badge */}
                  <span
                    className={`inline-flex items-center rounded-full px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider ${
                      status === 'running'
                        ? 'bg-cyan-500/20 text-cyan-300'
                        : status === 'completed'
                        ? 'bg-green-500/20 text-green-300'
                        : 'bg-slate-500/20 text-slate-400'
                    }`}
                  >
                    {status === 'running' && (
                      <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400" />
                    )}
                    {status}
                  </span>
                </div>
                <p className="mt-0.5 text-[10px] leading-snug text-slate-400 line-clamp-2">
                  {scenario.description}
                </p>
                {/* Expected tier badge */}
                <span
                  className={`mt-1 inline-block rounded border px-1.5 py-0.5 text-[9px] font-mono font-semibold ${tierClass}`}
                >
                  Expected: {scenario.expectedTier}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
