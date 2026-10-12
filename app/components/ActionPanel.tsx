'use client';

import { CRMAction, CRM_ACTION_LABELS, CRM_ACTION_RISK } from '@/app/lib/types';

interface ActionPanelProps {
  selectedAction: CRMAction;
  onActionChange: (action: CRMAction) => void;
  isActive: boolean;
}

/** Risk weight color: green for low, amber for medium, red for high */
function riskColor(weight: number): string {
  if (weight <= 0.3) return 'var(--green)';
  if (weight <= 0.6) return 'var(--amber)';
  return 'var(--red)';
}

function riskBgClass(weight: number): string {
  if (weight <= 0.3) return 'bg-green-500/10 border-green-500/30';
  if (weight <= 0.6) return 'bg-amber-500/10 border-amber-500/30';
  return 'bg-red-500/10 border-red-500/30';
}

export default function ActionPanel({
  selectedAction,
  onActionChange,
  isActive,
}: ActionPanelProps) {
  const actions = Object.keys(CRM_ACTION_LABELS) as CRMAction[];
  const selectedRisk = CRM_ACTION_RISK[selectedAction];

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
        <h3 className="text-xs font-semibold tracking-widest text-[var(--cyan)] uppercase">
          Banking Action Context
        </h3>
        <p className="mt-1 text-[11px] text-slate-400">
          Select the requested banking action to apply context risk weighting
        </p>
      </div>

      {/* Current risk weight display */}
      <div className="mb-3 flex items-center gap-2 rounded-lg border border-[var(--card-border)] bg-[var(--background)] px-3 py-2">
        <span className="text-xs text-slate-400">Current Risk Weight</span>
        <span className="ml-auto font-mono text-lg font-bold" style={{ color: riskColor(selectedRisk.weight) }}>
          {selectedRisk.weight.toFixed(2)}
        </span>
        <span
          className="inline-block h-3 w-3 rounded-full"
          style={{ backgroundColor: riskColor(selectedRisk.weight) }}
        />
      </div>

      {/* Action buttons grid */}
      <div className="grid grid-cols-2 gap-2">
        {actions.map((action) => {
          const label = CRM_ACTION_LABELS[action];
          const riskInfo = CRM_ACTION_RISK[action];
          const weight = riskInfo.weight;
          const isSelected = action === selectedAction;
          const color = riskColor(weight);

          return (
            <button
              key={action}
              onClick={() => onActionChange(action)}
              className={`group relative flex flex-col items-start rounded-lg border px-3 py-2 text-left transition-all duration-200 ${
                isSelected
                  ? `${riskBgClass(weight)} ring-1`
                  : 'border-[var(--card-border)] bg-[var(--background)] hover:border-slate-500'
              }`}
              style={
                isSelected
                  ? { borderColor: color }
                  : undefined
              }
            >
              {/* Risk indicator dot */}
              <span className="absolute right-2 top-2 flex items-center gap-1">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: color }}
                />
              </span>
              <span
                className={`text-xs font-medium leading-tight ${
                  isSelected ? 'text-white' : 'text-slate-300'
                }`}
              >
                {label}
              </span>
              <span className="mt-0.5 font-mono text-[10px] text-slate-500">
                w: {weight.toFixed(2)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
