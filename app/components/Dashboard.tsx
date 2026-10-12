'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  CRMAction,
  FusionResult,
  ScoreHistoryEntry,
  RiskTier,
} from '@/app/lib/types';
import WaveformVisualizer from '@/app/components/WaveformVisualizer';
import SpectrogramVisualizer from '@/app/components/SpectrogramVisualizer';
import RiskStrip from '@/app/components/RiskStrip';
import ScoreCard from '@/app/components/ScoreCard';
import ScoreTimeline from '@/app/components/ScoreTimeline';
import ReasonTags from '@/app/components/ReasonTags';
import ActionPanel from '@/app/components/ActionPanel';
import CallSimulator from '@/app/components/CallSimulator';
import AudioCapture from '@/app/components/AudioCapture';
import StepUpModal from '@/app/components/StepUpModal';
import OverallAnalysisModal from '@/app/components/OverallAnalysisModal';
import { generateOverallCallReport } from '@/app/lib/ml-pipeline';
import type { OverallCallReport } from '@/app/lib/ml-pipeline';
import type { TestScenario } from '@/app/components/CallSimulator';

const MAX_HISTORY = 100;

/** Default idle state */
const IDLE_RESULT: FusionResult = {
  finalScore: 0,
  tier: 'GREEN' as RiskTier,
  acoustic: {
    score: 0,
    features: {
      spectralFlatness: 0,
      zeroCrossingRate: 0,
      zeroCrossingRateVariance: 0,
      spectralCentroid: 0,
      spectralCentroidVariance: 0,
      spectralRolloff: 0,
      harmonicToNoiseRatio: 0,
      pitchContourVariance: 0,
      rmsEnergy: 0,
      spectralBandwidth: 0,
    },
    anomalies: [],
  },
  timing: {
    score: 0,
    averageResponseGap: 0,
    silenceRatio: 0,
    speechBurstRegularity: 0,
    totalSpeechDuration: 0,
    totalSilenceDuration: 0,
  },
  context: {
    score: 0,
    actionType: 'BALANCE_INQUIRY',
    riskWeight: 0.1,
    riskLabel: 'Low Risk',
  },
  reasonTags: [],
  timestamp: Date.now(),
  weights: { w1: 0.5, w2: 0.25, w3: 0.25 },
  latencyMs: 0,
};

export default function Dashboard() {
  // ── State ─────────────────────────────────────────────────────────
  const [currentResult, setCurrentResult] = useState<FusionResult>(IDLE_RESULT);
  const [scoreHistory, setScoreHistory] = useState<ScoreHistoryEntry[]>([]);
  const [selectedAction, setSelectedAction] = useState<CRMAction>('BALANCE_INQUIRY');
  const [isActive, setIsActive] = useState(false);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const [rmsLevel, setRmsLevel] = useState(0);
  const [isStepUpOpen, setIsStepUpOpen] = useState(false);
  const [stepUpStatus, setStepUpStatus] = useState<string | null>(null);
  const [isOverallOpen, setIsOverallOpen] = useState(false);
  const [overallReport, setOverallReport] = useState<OverallCallReport | null>(null);

  const resultCountRef = useRef(0);
  const callIdRef = useRef('call-849201');
  const scoreHistoryRef = useRef<ScoreHistoryEntry[]>([]);
  const currentResultRef = useRef<FusionResult>(IDLE_RESULT);
  const selectedActionRef = useRef<CRMAction>('BALANCE_INQUIRY');

  useEffect(() => {
    currentResultRef.current = currentResult;
  }, [currentResult]);

  useEffect(() => {
    selectedActionRef.current = selectedAction;
  }, [selectedAction]);

  useEffect(() => {
    callIdRef.current = `call-${Date.now().toString().slice(-6)}`;
  }, []);

  // ── Callbacks ─────────────────────────────────────────────────────
  const handleAnalysisResult = useCallback((result: FusionResult) => {
    setCurrentResult(result);
    currentResultRef.current = result;
    resultCountRef.current += 1;

    const entry: ScoreHistoryEntry = {
      timestamp: Date.now(),
      finalScore: result.finalScore,
      acousticScore: result.acoustic.score,
      timingScore: result.timing.score,
      contextScore: result.context.score,
      tier: result.tier,
    };
    setScoreHistory((prev) => {
      const next = [...prev, entry];
      const capped = next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
      scoreHistoryRef.current = capped;
      return capped;
    });
  }, []);

  const handleVisualizationData = useCallback(
    (data: { analyserNode: AnalyserNode | null; rmsLevel: number }) => {
      setAnalyserNode(data.analyserNode);
      setRmsLevel(data.rmsLevel);
    },
    []
  );

  const handleScenarioSelect = useCallback((scenario: TestScenario) => {
    setSelectedAction(scenario.actionType);
  }, []);

  const handleActiveChange = useCallback((active: boolean) => {
    setIsActive(active);
    if (!active) {
      resultCountRef.current = 0;
      setStepUpStatus(null);
    } else {
      callIdRef.current = `call-${Math.floor(100000 + Math.random() * 900000)}`;
      // Keep previous overallReport or clear if new active session starts
    }
  }, []);

  const handleStepUpResolution = (approved: boolean) => {
    if (approved) {
      setStepUpStatus('verified');
    } else {
      setStepUpStatus('fraud_blocked');
    }
  };

  const handleGenerateOverallReport = useCallback(async () => {
    const history = scoreHistoryRef.current.length > 0 ? scoreHistoryRef.current : scoreHistory;
    const currentRes = currentResultRef.current;
    const action = selectedActionRef.current;

    // If no history yet, synthesize a baseline frame history based on currentResult
    const historyToUse = history.length > 0 ? history : [
      {
        timestamp: Date.now() - 4000,
        finalScore: currentRes.finalScore,
        acousticScore: currentRes.acoustic.score,
        timingScore: currentRes.timing.score,
        contextScore: currentRes.context.score,
        tier: currentRes.tier,
      },
      {
        timestamp: Date.now() - 2000,
        finalScore: currentRes.finalScore,
        acousticScore: currentRes.acoustic.score,
        timingScore: currentRes.timing.score,
        contextScore: currentRes.context.score,
        tier: currentRes.tier,
      },
      {
        timestamp: Date.now(),
        finalScore: currentRes.finalScore,
        acousticScore: currentRes.acoustic.score,
        timingScore: currentRes.timing.score,
        contextScore: currentRes.context.score,
        tier: currentRes.tier,
      }
    ];

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'overall-call',
          callId: callIdRef.current,
          history: historyToUse,
          actionType: action,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.report) {
          setOverallReport(data.report);
          setIsOverallOpen(true);
          return;
        }
      }
    } catch {}

    // Fallback to internal TypeScript ML pipeline
    const report = generateOverallCallReport(callIdRef.current, historyToUse, action);
    setOverallReport(report);
    setIsOverallOpen(true);
  }, [scoreHistory]);

  const handleAudioComplete = useCallback(() => {
    // Automatically trigger custom ML pipeline for final overall call verdict
    setTimeout(() => {
      handleGenerateOverallReport();
    }, 150);
  }, [handleGenerateOverallReport]);

  // ── Derived ───────────────────────────────────────────────────────
  const { acoustic, timing, context, reasonTags, tier, finalScore, latencyMs } =
    currentResult;

  const acousticDetails = [
    { label: 'Spectral Flatness', value: acoustic.features.spectralFlatness.toFixed(3) },
    { label: 'ZCR Variance', value: acoustic.features.zeroCrossingRateVariance.toFixed(4) },
    { label: 'HNR', value: `${acoustic.features.harmonicToNoiseRatio.toFixed(1)} dB` },
    { label: 'Pitch Variance', value: acoustic.features.pitchContourVariance.toFixed(1) },
  ];

  const timingDetails = [
    { label: 'Avg Response Gap', value: `${timing.averageResponseGap.toFixed(0)} ms` },
    { label: 'Silence Ratio', value: `${(timing.silenceRatio * 100).toFixed(1)}%` },
    { label: 'Speech Regularity', value: timing.speechBurstRegularity.toFixed(3) },
  ];

  const contextDetails = [
    { label: 'Action Type', value: context.actionType.replace(/_/g, ' ') },
    { label: 'Risk Weight', value: context.riskWeight.toFixed(2) },
    { label: 'Risk Level', value: context.riskLabel },
  ];

  // ── Render ────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col min-h-screen">
      {/* ━━ Header ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <header className="flex items-center justify-between border-b border-[var(--card-border)] bg-[var(--card-bg)] px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-[var(--cyan)]">भरोसा</span>
            <span className="text-lg font-semibold text-white">Bharosa</span>
          </div>
          <span className="hidden sm:inline-block text-xs text-slate-400 border-l border-[var(--card-border)] pl-3 ml-1">
            Voice Risk Co-Pilot for Digital Banking • RAKSHAM 2026
          </span>
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-400">
          {/* Overall Forensic Report Button */}
          <button
            onClick={handleGenerateOverallReport}
            className="flex items-center gap-1.5 rounded-lg border border-[var(--cyan)]/40 bg-[var(--cyan)]/15 px-3 py-1 font-bold text-[var(--cyan)] hover:bg-[var(--cyan)]/25 transition shadow-sm"
          >
            <span>📊</span> Overall Forensic Report
          </button>

          {/* Out-of-band trigger button */}
          {(tier === 'RED' || tier === 'AMBER') && isActive && (
            <button
              onClick={() => setIsStepUpOpen(true)}
              className="flex items-center gap-1.5 rounded-lg border border-red-500/40 bg-red-500/20 px-3 py-1 font-bold text-red-300 hover:bg-red-500/30 transition animate-pulse"
            >
              <span>📱</span> In-App Push
            </button>
          )}

          {/* Connection status */}
          <div className="flex items-center gap-1.5 border-l border-slate-800 pl-3">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                isActive ? 'bg-green-400 animate-pulse' : 'bg-slate-500'
              }`}
            />
            <span className="hidden md:inline">{isActive ? 'SIPREC ACTIVE' : 'STANDBY'}</span>
          </div>
          {isActive && (
            <span className="font-mono text-[var(--cyan)]">
              {latencyMs.toFixed(0)}ms
            </span>
          )}
        </div>
      </header>

      {/* ━━ Risk Strip ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <RiskStrip tier={tier} score={finalScore} isActive={isActive} latencyMs={latencyMs} />

      {/* ━━ Agent Script & Friction Advisory Bar ━━━━━━━━━━━━━━━━━━━━ */}
      {isActive && (
        <div className={`px-6 py-2 border-b text-xs flex items-center justify-between transition-colors ${
          tier === 'RED'
            ? 'bg-red-950/40 border-red-900/60 text-red-200'
            : tier === 'AMBER'
            ? 'bg-amber-950/30 border-amber-900/50 text-amber-200'
            : 'bg-green-950/20 border-green-900/40 text-green-200'
        }`}>
          <div className="flex items-center gap-2">
            <span className="font-bold uppercase tracking-wider text-[10px] px-1.5 py-0.5 rounded bg-black/40">
              Agent Script Guidance:
            </span>
            <span className="italic">
              {tier === 'RED'
                ? '"I understand your request. For your account security, I have initiated an authorization challenge to your registered mobile app."'
                : tier === 'AMBER'
                ? '"Please hold briefly while I send a 20-second verification prompt to your banking app."'
                : '"Identity verified naturally. How may I assist you with your banking needs today?"'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {stepUpStatus === 'verified' && (
              <span className="font-bold text-green-400 flex items-center gap-1">
                ✓ In-App Passkey Confirmed
              </span>
            )}
            {stepUpStatus === 'fraud_blocked' && (
              <span className="font-bold text-red-400 flex items-center gap-1">
                🚫 Action Locked (Incident Logged)
              </span>
            )}
            <button
              onClick={() => setIsStepUpOpen(true)}
              className="text-[11px] underline hover:text-white"
            >
              Open Step-Up Console
            </button>
          </div>
        </div>
      )}

      {/* ━━ Final ML Pipeline Evaluation Verdict Banner ━━━━━━━━━━━━━━ */}
      {overallReport && !isActive && (
        <div className={`mx-4 my-2 px-5 py-3 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-lg transition animate-fade-slide-in ${
          overallReport.overallVerdict === 'SYNTHETIC_VOICE_CLONE_DETECTED'
            ? 'bg-red-950/70 border-red-500/60 text-red-100 glow-red'
            : overallReport.overallVerdict === 'SUSPICIOUS_ACOUSTIC_ANOMALIES'
            ? 'bg-amber-950/60 border-amber-500/60 text-amber-100 glow-amber'
            : 'bg-green-950/60 border-green-500/60 text-green-100 glow-green'
        }`}>
          <div className="flex items-center gap-3.5">
            <span className="text-3xl">
              {overallReport.overallVerdict === 'SYNTHETIC_VOICE_CLONE_DETECTED'
                ? '🚨'
                : overallReport.overallVerdict === 'SUSPICIOUS_ACOUSTIC_ANOMALIES'
                ? '⚠️'
                : '🛡️'}
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded bg-black/50 border border-white/10">
                  Definitive Overall Call Verdict
                </span>
                <span className="font-extrabold text-sm sm:text-base tracking-wide">
                  {overallReport.overallVerdict === 'SYNTHETIC_VOICE_CLONE_DETECTED'
                    ? 'SYNTHETIC AI VOICE CLONE DETECTED'
                    : overallReport.overallVerdict === 'SUSPICIOUS_ACOUSTIC_ANOMALIES'
                    ? 'SUSPICIOUS ACOUSTIC DISCONTINUITIES'
                    : 'AUTHENTIC NATURAL HUMAN PRESENCE'}
                </span>
                <span className={`rounded px-2 py-0.5 font-mono text-xs font-bold ${
                  overallReport.overallVerdict === 'SYNTHETIC_VOICE_CLONE_DETECTED'
                    ? 'bg-red-500 text-white'
                    : overallReport.overallVerdict === 'SUSPICIOUS_ACOUSTIC_ANOMALIES'
                    ? 'bg-amber-500 text-black'
                    : 'bg-green-500 text-black'
                }`}>
                  {overallReport.overallVerdict === 'SYNTHETIC_VOICE_CLONE_DETECTED'
                    ? `${overallReport.cumulativeAiProbability}% AI Prob.`
                    : `${overallReport.humanConfidenceScore}% Human Conf.`}
                </span>
              </div>
              <p className="text-xs text-slate-300 mt-1 max-w-2xl">
                {overallReport.operationalRecommendation}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 self-end sm:self-auto">
            <button
              onClick={() => setIsOverallOpen(true)}
              className="rounded-lg bg-white/15 hover:bg-white/25 border border-white/20 px-3.5 py-1.5 text-xs font-bold text-white transition flex items-center gap-1.5 whitespace-nowrap shadow-sm"
            >
              <span>🏛️</span> Full Forensic Dossier
            </button>
            <button
              onClick={() => setOverallReport(null)}
              className="rounded-lg bg-black/40 hover:bg-black/60 px-2.5 py-1.5 text-xs text-slate-400 hover:text-white transition"
              title="Dismiss"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* ━━ Main Content ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-4 p-4 overflow-auto">
        {/* ── Left Column ──────────────────────────────────────────── */}
        <div className="flex flex-col gap-4 min-w-0">
          {/* Audio capture controls */}
          <AudioCapture
            onAnalysisResult={handleAnalysisResult}
            onVisualizationData={handleVisualizationData}
            actionType={selectedAction}
            isActive={isActive}
            onActiveChange={handleActiveChange}
            onTriggerStepUp={() => setIsStepUpOpen(true)}
            onAudioComplete={handleAudioComplete}
          />

          {/* Waveform + Spectrogram */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <WaveformVisualizer
              analyserNode={analyserNode}
              isActive={isActive}
              rmsLevel={rmsLevel}
            />
            <SpectrogramVisualizer
              analyserNode={analyserNode}
              isActive={isActive}
            />
          </div>

          {/* Score Timeline */}
          <ScoreTimeline history={scoreHistory} maxPoints={MAX_HISTORY} />
        </div>

        {/* ── Right Column ─────────────────────────────────────────── */}
        <div className="flex flex-col gap-4">
          {/* Three Score Cards */}
          <ScoreCard
            title="Acoustic Signal"
            score={acoustic.score}
            maxScore={1}
            icon="🔊"
            details={acousticDetails}
            color="var(--cyan)"
            signal="acoustic"
          />
          <ScoreCard
            title="Timing Signal"
            score={timing.score}
            maxScore={1}
            icon="⏱️"
            details={timingDetails}
            color="var(--purple)"
            signal="timing"
          />
          <ScoreCard
            title="Context Signal"
            score={context.score}
            maxScore={1}
            icon="🏦"
            details={contextDetails}
            color="var(--amber)"
            signal="context"
          />

          {/* Reason Tags */}
          <ReasonTags tags={reasonTags} isActive={isActive} />

          {/* Action Panel */}
          <ActionPanel
            selectedAction={selectedAction}
            onActionChange={setSelectedAction}
            isActive={!isActive}
          />

          {/* Call Simulator */}
          <CallSimulator
            onScenarioSelect={handleScenarioSelect}
            isActive={!isActive}
          />
        </div>
      </main>

      {/* ━━ Step-Up Modal ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <StepUpModal
        isOpen={isStepUpOpen}
        onClose={() => setIsStepUpOpen(false)}
        result={currentResult}
        actionType={selectedAction}
        onStepUpResolution={handleStepUpResolution}
      />

      {/* ━━ Overall Post-Call Forensic Analysis Modal ━━━━━━━━━━━━━━ */}
      <OverallAnalysisModal
        isOpen={isOverallOpen}
        onClose={() => setIsOverallOpen(false)}
        report={overallReport}
      />

      {/* ━━ Footer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <footer className="border-t border-[var(--card-border)] bg-[var(--card-bg)] px-6 py-2 text-center">
        <p className="text-[10px] tracking-wider text-slate-500">
          Ephemeral Processing • Zero Voiceprints Stored • DPDP Act 2023 Compliant • Sub-200ms Decision Engine
        </p>
      </footer>
    </div>
  );
}
