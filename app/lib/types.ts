// Bharosa Voice Risk Co-Pilot - Core Type Definitions

export type RiskTier = 'GREEN' | 'AMBER' | 'RED';

export type CRMAction =
  | 'BALANCE_INQUIRY'
  | 'CARD_BLOCK'
  | 'PIN_RESET'
  | 'SIM_CHANGE'
  | 'BENEFICIARY_ADD'
  | 'ADDRESS_CHANGE'
  | 'ACCOUNT_CLOSURE'
  | 'FUND_TRANSFER';

export interface AcousticFeatures {
  spectralFlatness: number;
  zeroCrossingRate: number;
  zeroCrossingRateVariance: number;
  spectralCentroid: number;
  spectralCentroidVariance: number;
  spectralRolloff: number;
  harmonicToNoiseRatio: number;
  pitchContourVariance: number;
  rmsEnergy: number;
  spectralBandwidth: number;
}

export interface AcousticScore {
  score: number;
  features: AcousticFeatures;
  anomalies: string[];
}

export interface TimingScore {
  score: number;
  averageResponseGap: number;
  silenceRatio: number;
  speechBurstRegularity: number;
  totalSpeechDuration: number;
  totalSilenceDuration: number;
}

export interface ContextScore {
  score: number;
  actionType: CRMAction;
  riskWeight: number;
  riskLabel: string;
}

export interface ReasonTag {
  code: string;
  label: string;
  severity: 'low' | 'medium' | 'high';
  signal: 'acoustic' | 'timing' | 'context';
}

export interface FusionResult {
  finalScore: number;
  tier: RiskTier;
  acoustic: AcousticScore;
  timing: TimingScore;
  context: ContextScore;
  reasonTags: ReasonTag[];
  timestamp: number;
  weights: { w1: number; w2: number; w3: number };
  latencyMs: number;
}

export interface AnalysisRequest {
  audioData: Float32Array;
  sampleRate: number;
  windowDuration: number;
  actionType: CRMAction;
  callDuration: number;
  speechSegments: SpeechSegment[];
}

export interface SpeechSegment {
  start: number;
  end: number;
  isSpeech: boolean;
}

export interface ScoreHistoryEntry {
  timestamp: number;
  finalScore: number;
  tier: RiskTier;
  acousticScore: number;
  timingScore: number;
  contextScore: number;
}

export interface CallSession {
  id: string;
  startTime: number;
  duration: number;
  actionType: CRMAction;
  isActive: boolean;
  scoreHistory: ScoreHistoryEntry[];
  currentResult: FusionResult | null;
}

export const CRM_ACTION_LABELS: Record<CRMAction, string> = {
  BALANCE_INQUIRY: 'Balance Inquiry',
  CARD_BLOCK: 'Card Block',
  PIN_RESET: 'PIN Reset',
  SIM_CHANGE: 'SIM/Mobile Change',
  BENEFICIARY_ADD: 'Beneficiary Addition',
  ADDRESS_CHANGE: 'Address Change',
  ACCOUNT_CLOSURE: 'Account Closure',
  FUND_TRANSFER: 'Fund Transfer',
};

export const CRM_ACTION_RISK: Record<CRMAction, { weight: number; label: string }> = {
  BALANCE_INQUIRY: { weight: 0.1, label: 'Low Risk' },
  CARD_BLOCK: { weight: 0.3, label: 'Low-Medium Risk' },
  PIN_RESET: { weight: 0.5, label: 'Medium Risk' },
  ADDRESS_CHANGE: { weight: 0.4, label: 'Medium Risk' },
  SIM_CHANGE: { weight: 0.9, label: 'Critical Risk' },
  BENEFICIARY_ADD: { weight: 0.8, label: 'High Risk' },
  ACCOUNT_CLOSURE: { weight: 0.7, label: 'High Risk' },
  FUND_TRANSFER: { weight: 0.85, label: 'High Risk' },
};

export const TIER_THRESHOLDS = {
  GREEN_MAX: 0.30,
  AMBER_MAX: 0.65,
} as const;

export const FUSION_WEIGHTS = {
  w1: 0.50,  // Acoustic weight
  w2: 0.25,  // Timing weight
  w3: 0.25,  // Context weight
} as const;
