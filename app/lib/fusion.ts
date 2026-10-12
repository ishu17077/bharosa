// Bharosa Voice Risk Co-Pilot - Calibrated Fusion Engine
// Combines tri-signal scores into a single risk assessment

import type {
  FusionResult,
  AcousticScore,
  TimingScore,
  ContextScore,
  RiskTier,
} from './types';
import { TIER_THRESHOLDS, FUSION_WEIGHTS } from './types';
import { generateReasonTags } from './scoring';

// ─── Utility ─────────────────────────────────────────────────────────────────

/**
 * Sigmoid activation function.
 * Maps any real value to the range (0, 1).
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Classify a final score into a risk tier.
 *
 * GREEN: score ≤ 0.30 — Low risk, proceed normally
 * AMBER: score ≤ 0.65 — Moderate risk, additional verification recommended
 * RED:   score > 0.65  — High risk, escalate or block
 */
function classifyTier(score: number): RiskTier {
  if (score <= TIER_THRESHOLDS.GREEN_MAX) return 'GREEN';
  if (score <= TIER_THRESHOLDS.AMBER_MAX) return 'AMBER';
  return 'RED';
}

// ─── Main Fusion ─────────────────────────────────────────────────────────────

/**
 * Fuse all three signal scores into a calibrated final risk score.
 *
 * Formula: S_final = σ(w1·A + w2·T + w3·C)
 *
 * Where:
 *   σ = sigmoid activation
 *   w1 = 0.50 (acoustic weight — primary signal)
 *   w2 = 0.25 (timing weight — secondary signal)
 *   w3 = 0.25 (context weight — risk multiplier)
 *   A, T, C = individual signal scores mapped to logit space
 *
 * The individual scores are first mapped from [0,1] to logit space
 * via inverse sigmoid, then the weighted sum is mapped back through
 * sigmoid to produce the final calibrated score.
 *
 * @param acoustic - Acoustic analysis score
 * @param timing - Timing analysis score
 * @param context - CRM context risk score
 * @returns Complete FusionResult with tier classification and reason tags
 */
export function fuseScores(
  acoustic: AcousticScore,
  timing: TimingScore,
  context: ContextScore
): FusionResult {
  const startTime = performance.now();

  const { w1, w2, w3 } = FUSION_WEIGHTS;

  // Map individual scores [0,1] to logit space for proper linear combination
  // Clamp to avoid ±Infinity from logit of 0 or 1
  const clampedA = Math.max(0.001, Math.min(0.999, acoustic.score));
  const clampedT = Math.max(0.001, Math.min(0.999, timing.score));
  const clampedC = Math.max(0.001, Math.min(0.999, context.score));

  const logitA = Math.log(clampedA / (1 - clampedA));
  const logitT = Math.log(clampedT / (1 - clampedT));
  const logitC = Math.log(clampedC / (1 - clampedC));

  // Weighted sum in logit space
  const weightedLogit = w1 * logitA + w2 * logitT + w3 * logitC;

  // Map back through sigmoid to calibrated [0,1] score
  const finalScore = sigmoid(weightedLogit);

  // Classify tier
  const tier = classifyTier(finalScore);

  // Generate reason tags
  const reasonTags = generateReasonTags(acoustic, timing, context);

  const latencyMs = performance.now() - startTime;

  return {
    finalScore,
    tier,
    acoustic,
    timing,
    context,
    reasonTags,
    timestamp: Date.now(),
    weights: { w1, w2, w3 },
    latencyMs,
  };
}
