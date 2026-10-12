// Bharosa Voice Risk Co-Pilot - Tri-Signal Scoring Engine
// Scores acoustic features, conversational timing, and CRM context risk

import type {
  AcousticFeatures,
  AcousticScore,
  TimingScore,
  ContextScore,
  SpeechSegment,
  CRMAction,
  ReasonTag,
} from './types';
import { CRM_ACTION_RISK } from './types';

// ─── Utility ─────────────────────────────────────────────────────────────────

/**
 * Sigmoid activation for smooth threshold mapping.
 * Maps any real value to [0, 1].
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Clamp a value to [0, 1].
 */
function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

// ─── Acoustic Scoring ───────────────────────────────────────────────────────

/**
 * Score acoustic features to detect synthetic voice anomalies.
 *
 * Synthetic voices typically exhibit:
 * - Higher spectral flatness (unnaturally smooth spectrum, natural ≈ 0.1-0.3)
 * - Lower ZCR variance (too regular timing patterns)
 * - Lower spectral centroid variance (monotonous spectral content)
 * - Lower HNR (phase artifacts from vocoders)
 * - Lower pitch variance (unnatural, robotic prosody)
 *
 * Each sub-score is mapped through a sigmoid with calibrated thresholds,
 * then combined with sub-weights into a final acoustic anomaly score [0, 1].
 */
export function scoreAcoustic(features: AcousticFeatures): AcousticScore {
  const anomalies: string[] = [];

  // ── Sub-score 1: Spectral Flatness ──
  // Natural speech: ~0.05 - 0.25, Synthetic: 0.35+
  // Higher flatness → higher anomaly score
  const flatnessScore = sigmoid((features.spectralFlatness - 0.25) * 12);
  if (features.spectralFlatness > 0.4) {
    anomalies.push('Unnatural Spectral Smoothness');
  }

  // ── Sub-score 2: ZCR Variance ──
  // Natural speech has high ZCR variance due to phoneme transitions
  // Low variance → synthetic (too regular)
  const zcrVarScore = sigmoid((0.008 - features.zeroCrossingRateVariance) * 300);
  if (features.zeroCrossingRateVariance < 0.005) {
    anomalies.push('Abnormally Regular Zero-Crossing Pattern');
  }

  // ── Sub-score 3: Spectral Centroid Variance ──
  // Natural speech has variable brightness; synthetic is more stable
  // Normalized variance: low → suspicious
  const centroidVarScore = sigmoid((0.02 - features.spectralCentroidVariance) * 80);
  if (features.spectralCentroidVariance < 0.01) {
    anomalies.push('Monotonous Spectral Content');
  }

  // ── Sub-score 4: Harmonic-to-Noise Ratio ──
  // Natural voiced speech HNR: 10-25 dB
  // Vocoder artifacts reduce HNR below ~8 dB
  // Very low HNR → likely synthetic with phase discontinuities
  const hnrScore = sigmoid((8 - features.harmonicToNoiseRatio) * 0.6);
  if (features.harmonicToNoiseRatio < 5) {
    anomalies.push('Phase Discontinuity Detected');
  }
  if (features.harmonicToNoiseRatio < 3) {
    anomalies.push('Severe Vocoder Artifacts');
  }

  // ── Sub-score 5: Pitch Contour Variance ──
  // Natural speech has expressive pitch variation
  // Very low pitch variance → robotic monotone
  // Normalized pitch variance (coefficient of variation squared): natural ~0.01-0.1
  const pitchVarScore = sigmoid((0.005 - features.pitchContourVariance) * 200);
  if (features.pitchContourVariance < 0.003) {
    anomalies.push('Monotonous Pitch Pattern');
  }

  // ── Sub-score 6: Spectral Rolloff anomaly ──
  // Extremely low or high rolloff can indicate bandwidth limiting or synthesis
  const rolloffNorm = features.spectralRolloff / 8000; // Normalize to ~Nyquist of phone audio
  const rolloffScore = rolloffNorm < 0.2 || rolloffNorm > 0.95
    ? sigmoid((rolloffNorm < 0.2 ? (0.2 - rolloffNorm) : (rolloffNorm - 0.95)) * 20)
    : 0;
  if (rolloffNorm < 0.15) {
    anomalies.push('Abnormal Bandwidth Limitation');
  }

  // ── Combine sub-scores with weights ──
  const combinedScore =
    0.25 * flatnessScore +
    0.15 * zcrVarScore +
    0.15 * centroidVarScore +
    0.20 * hnrScore +
    0.15 * pitchVarScore +
    0.10 * rolloffScore;

  return {
    score: clamp01(combinedScore),
    features,
    anomalies,
  };
}

// ─── Timing Scoring ─────────────────────────────────────────────────────────

/**
 * Score conversational timing to detect generative lag.
 *
 * Synthetic real-time systems typically show:
 * - Higher response gaps (LLM generation latency, typically 200-800ms)
 * - More regular speech burst patterns (deterministic scheduling)
 * - Unnatural silence ratios (processing pauses)
 *
 * Analyzes speech segments to extract timing patterns and scores
 * deviation from natural human conversation dynamics.
 */
export function scoreTiming(segments: SpeechSegment[], callDuration: number): TimingScore {
  if (segments.length === 0 || callDuration <= 0) {
    return {
      score: 0,
      averageResponseGap: 0,
      silenceRatio: 0,
      speechBurstRegularity: 0,
      totalSpeechDuration: 0,
      totalSilenceDuration: 0,
    };
  }

  // ── Compute total speech and silence durations ──
  let totalSpeechDuration = 0;
  let totalSilenceDuration = 0;

  for (const seg of segments) {
    const duration = seg.end - seg.start;
    if (seg.isSpeech) {
      totalSpeechDuration += duration;
    } else {
      totalSilenceDuration += duration;
    }
  }

  // ── Compute response gaps (silence segments between speech bursts) ──
  const silenceGaps: number[] = [];
  for (const seg of segments) {
    if (!seg.isSpeech) {
      const duration = seg.end - seg.start;
      if (duration > 0.05) { // Ignore very short pauses < 50ms
        silenceGaps.push(duration);
      }
    }
  }

  const averageResponseGap = silenceGaps.length > 0
    ? silenceGaps.reduce((a, b) => a + b, 0) / silenceGaps.length
    : 0;

  // ── Compute speech burst regularity ──
  // Standard deviation of speech burst durations, normalized by mean
  // Low coefficient of variation → suspiciously regular bursts
  const speechBursts: number[] = [];
  for (const seg of segments) {
    if (seg.isSpeech) {
      speechBursts.push(seg.end - seg.start);
    }
  }

  let speechBurstRegularity = 0;
  if (speechBursts.length >= 2) {
    const mean = speechBursts.reduce((a, b) => a + b, 0) / speechBursts.length;
    if (mean > 0.01) {
      const variance = speechBursts.reduce((sum, b) => sum + (b - mean) * (b - mean), 0) / speechBursts.length;
      const cv = Math.sqrt(variance) / mean; // Coefficient of variation
      // Low CV means very regular bursts → suspicious
      speechBurstRegularity = clamp01(1 - cv); // Higher regularity → higher score
    }
  }

  // ── Silence ratio ──
  const silenceRatio = callDuration > 0
    ? totalSilenceDuration / callDuration
    : 0;

  // ── Score sub-components ──

  // Response gap score: gaps > 300ms are suspicious for AI generation latency
  // Natural human response time: 150-250ms; AI systems: 300-800ms
  const gapScore = sigmoid((averageResponseGap - 0.25) * 8);

  // Silence ratio score: unnatural ratios (too low or too high)
  // Natural conversation: ~30-50% silence
  const silenceDeviation = Math.abs(silenceRatio - 0.4);
  const silenceScore = sigmoid((silenceDeviation - 0.15) * 8);

  // Regularity score: high regularity is suspicious
  const regularityScore = sigmoid((speechBurstRegularity - 0.5) * 6);

  // ── Combine timing sub-scores ──
  const combinedScore =
    0.45 * gapScore +
    0.25 * silenceScore +
    0.30 * regularityScore;

  return {
    score: clamp01(combinedScore),
    averageResponseGap,
    silenceRatio,
    speechBurstRegularity,
    totalSpeechDuration,
    totalSilenceDuration,
  };
}

// ─── Context Scoring ────────────────────────────────────────────────────────

/**
 * Score CRM context risk based on the banking action being performed.
 *
 * Higher-risk actions (SIM change, beneficiary addition, fund transfer)
 * warrant lower tolerance for voice anomalies.
 */
export function scoreContext(actionType: CRMAction): ContextScore {
  const riskInfo = CRM_ACTION_RISK[actionType];

  return {
    score: riskInfo.weight,
    actionType,
    riskWeight: riskInfo.weight,
    riskLabel: riskInfo.label,
  };
}

// ─── Reason Tag Generation ──────────────────────────────────────────────────

/**
 * Generate human-readable reason tags from all three signal scores.
 * These tags explain WHY a particular risk level was assigned,
 * enabling the agent to make informed decisions.
 */
export function generateReasonTags(
  acoustic: AcousticScore,
  timing: TimingScore,
  context: ContextScore
): ReasonTag[] {
  const tags: ReasonTag[] = [];

  // ── Acoustic reason tags ──
  for (const anomaly of acoustic.anomalies) {
    let severity: ReasonTag['severity'] = 'low';
    let code = 'ACO_GENERIC';

    switch (anomaly) {
      case 'Unnatural Spectral Smoothness':
        code = 'ACO_FLAT_SPECTRUM';
        severity = acoustic.features.spectralFlatness > 0.6 ? 'high' : 'medium';
        break;
      case 'Phase Discontinuity Detected':
        code = 'ACO_PHASE_DISC';
        severity = 'high';
        break;
      case 'Severe Vocoder Artifacts':
        code = 'ACO_VOCODER';
        severity = 'high';
        break;
      case 'Monotonous Pitch Pattern':
        code = 'ACO_MONO_PITCH';
        severity = acoustic.features.pitchContourVariance < 0.001 ? 'high' : 'medium';
        break;
      case 'Abnormally Regular Zero-Crossing Pattern':
        code = 'ACO_REG_ZCR';
        severity = 'medium';
        break;
      case 'Monotonous Spectral Content':
        code = 'ACO_MONO_SPEC';
        severity = 'medium';
        break;
      case 'Abnormal Bandwidth Limitation':
        code = 'ACO_BW_LIMIT';
        severity = 'medium';
        break;
    }

    tags.push({
      code,
      label: anomaly,
      severity,
      signal: 'acoustic',
    });
  }

  // ── Timing reason tags ──
  if (timing.averageResponseGap > 0.4) {
    tags.push({
      code: 'TIM_RESP_LAG',
      label: 'Suspicious Response Latency',
      severity: timing.averageResponseGap > 0.6 ? 'high' : 'medium',
      signal: 'timing',
    });
  }

  if (timing.speechBurstRegularity > 0.7) {
    tags.push({
      code: 'TIM_REG_BURST',
      label: 'Unnaturally Regular Speech Bursts',
      severity: timing.speechBurstRegularity > 0.85 ? 'high' : 'medium',
      signal: 'timing',
    });
  }

  if (timing.silenceRatio < 0.15 || timing.silenceRatio > 0.65) {
    tags.push({
      code: 'TIM_SIL_RATIO',
      label: 'Abnormal Silence Pattern',
      severity: 'medium',
      signal: 'timing',
    });
  }

  // ── Context reason tags ──
  if (context.riskWeight >= 0.7) {
    tags.push({
      code: 'CTX_HIGH_RISK',
      label: 'High-Risk Banking Action',
      severity: context.riskWeight >= 0.85 ? 'high' : 'medium',
      signal: 'context',
    });
  }

  if (context.riskWeight >= 0.9) {
    tags.push({
      code: 'CTX_CRITICAL',
      label: 'Critical Security Operation',
      severity: 'high',
      signal: 'context',
    });
  }

  return tags;
}
