'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { CRMAction } from '@/app/lib/types';
import type { FusionResult, SpeechSegment } from '@/app/lib/types';
import { analyzeAudioFrame } from '@/app/lib/dsp';
import { scoreAcoustic, scoreTiming, scoreContext } from '@/app/lib/scoring';
import { fuseScores } from '@/app/lib/fusion';
import { AudioProcessor } from '@/app/lib/audio-processor';
import {
  generateNaturalSpeechBuffer,
  generateClonedVoiceBuffer,
  generateNoisyCallerBuffer,
} from '@/app/lib/synthetic-benchmarks';

interface AudioCaptureProps {
  onAnalysisResult: (result: FusionResult) => void;
  onVisualizationData: (data: {
    analyserNode: AnalyserNode | null;
    rmsLevel: number;
  }) => void;
  actionType: CRMAction;
  isActive: boolean;
  onActiveChange: (active: boolean) => void;
  onTriggerStepUp?: () => void;
  onAudioComplete?: () => void;
}

type CaptureSource = 'mic' | 'benchmark' | 'file';
type CaptureStatus = 'idle' | 'listening' | 'analyzing';
type MicPermission = 'unknown' | 'granted' | 'denied' | 'prompt';

export default function AudioCapture({
  onAnalysisResult,
  onVisualizationData,
  actionType,
  isActive,
  onActiveChange,
  onTriggerStepUp,
  onAudioComplete,
}: AudioCaptureProps) {
  const [sourceMode, setSourceMode] = useState<CaptureSource>('mic');
  const [status, setStatus] = useState<CaptureStatus>('idle');
  const [micPermission, setMicPermission] = useState<MicPermission>('unknown');
  const [elapsed, setElapsed] = useState(0);
  const [activeBenchmarkName, setActiveBenchmarkName] = useState<string>('');
  const [playbackProgress, setPlaybackProgress] = useState(0);
  const [playbackDuration, setPlaybackDuration] = useState(0);
  const [playbackCurrentTime, setPlaybackCurrentTime] = useState(0);

  const processorRef = useRef<AudioProcessor | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const benchmarkTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startTimeRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const actionTypeRef = useRef<CRMAction>(actionType);
  useEffect(() => {
    actionTypeRef.current = actionType;
  }, [actionType]);

  // Check microphone permission on mount
  useEffect(() => {
    if (typeof navigator !== 'undefined' && navigator.permissions) {
      navigator.permissions
        .query({ name: 'microphone' as PermissionName })
        .then((result) => {
          setMicPermission(result.state as MicPermission);
          result.onchange = () =>
            setMicPermission(result.state as MicPermission);
        })
        .catch(() => setMicPermission('unknown'));
    }
  }, []);

  // Elapsed timer
  useEffect(() => {
    if (isActive) {
      startTimeRef.current = Date.now();
      timerRef.current = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }, 250);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setElapsed(0);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isActive]);

  const dispatchScoreToServer = useCallback(
    async (result: FusionResult) => {
      try {
        await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            callId: `call-${startTimeRef.current}`,
            timestamp: Date.now(),
            acousticScore: result.acoustic.score,
            timingScore: result.timing.score,
            actionType: actionTypeRef.current,
          }),
        });
      } catch {
        // Non-blocking server audit log
      }
    },
    []
  );

  // ── Mic Start ────────────────────────────────────────────────────────
  const handleStartMic = async () => {
    try {
      const processor = new AudioProcessor();
      startTimeRef.current = Date.now();

      await processor.start((data) => {
        setStatus('analyzing');

        const features = analyzeAudioFrame(data.audioData, data.sampleRate);
        const callDuration = (Date.now() - startTimeRef.current) / 1000;
        const acoustic = scoreAcoustic(features);
        const timing = scoreTiming(data.speechSegments, callDuration);
        const context = scoreContext(actionTypeRef.current);
        const result = fuseScores(acoustic, timing, context);

        onAnalysisResult(result);
        onVisualizationData({
          analyserNode: processor.getAnalyserNode(),
          rmsLevel: data.rmsLevel,
        });

        if (result.tier === 'RED' && onTriggerStepUp) {
          onTriggerStepUp();
        }

        dispatchScoreToServer(result);
        setStatus('listening');
      });

      processorRef.current = processor;
      setStatus('listening');
      setMicPermission('granted');
      onActiveChange(true);
    } catch {
      setMicPermission('denied');
      setStatus('idle');
    }
  };

  // ── Play Synthetic or File AudioBuffer ─────────────────────────────────
  const playAudioBufferStream = (
    audioBuffer: AudioBuffer,
    benchmarkLabel: string
  ) => {
    handleStop();

    const ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    audioContextRef.current = ctx;

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;
    analyserNodeRef.current = analyser;

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.loop = false; // Never loop uploaded audio or test benchmarks
    source.connect(analyser);
    analyser.connect(ctx.destination);
    sourceNodeRef.current = source;

    const totalDurationSec = audioBuffer.duration;
    setPlaybackDuration(totalDurationSec);
    setPlaybackCurrentTime(0);
    setPlaybackProgress(0);

    startTimeRef.current = Date.now();
    setActiveBenchmarkName(benchmarkLabel);
    onActiveChange(true);
    setStatus('listening');

    source.start(0);

    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    const frameSize = Math.floor(sampleRate * 0.8); // 800ms window
    const hopSize = Math.floor(sampleRate * 0.2); // 200ms hop
    let cursor = 0;
    const simulatedSegments: SpeechSegment[] = [];

    let isCompleted = false;
    const handleStreamComplete = () => {
      if (isCompleted) return;
      isCompleted = true;

      if (benchmarkTimerRef.current) {
        clearInterval(benchmarkTimerRef.current);
        benchmarkTimerRef.current = null;
      }
      if (sourceNodeRef.current) {
        try {
          sourceNodeRef.current.stop();
        } catch {}
        sourceNodeRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
        audioContextRef.current = null;
      }
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      setPlaybackProgress(100);
      setPlaybackCurrentTime(totalDurationSec);
      setStatus('idle');
      onActiveChange(false);
      onVisualizationData({ analyserNode: null, rmsLevel: 0 });

      // Trigger automatic final ML evaluation
      if (onAudioComplete) {
        onAudioComplete();
      }
    };

    source.onended = () => {
      handleStreamComplete();
    };

    benchmarkTimerRef.current = setInterval(() => {
      if (cursor + frameSize >= channelData.length) {
        // Audio stream reached end — cleanly finish and evaluate final ML verdict!
        handleStreamComplete();
        return;
      }

      const frame = new Float32Array(frameSize);
      for (let i = 0; i < frameSize; i++) {
        frame[i] = channelData[cursor + i] || 0;
      }
      cursor += hopSize;

      // Update playback progress
      const currentSec = Math.min(totalDurationSec, cursor / sampleRate);
      setPlaybackCurrentTime(currentSec);
      setPlaybackProgress(Math.min(100, Math.round((currentSec / totalDurationSec) * 100)));

      // Energy check
      let sumSq = 0;
      for (let i = 0; i < frame.length; i++) sumSq += frame[i] * frame[i];
      const rms = Math.sqrt(sumSq / frame.length);

      const elapsedSec = (Date.now() - startTimeRef.current) / 1000;
      if (rms > 0.01) {
        simulatedSegments.push({
          start: Math.max(0, elapsedSec - 0.2),
          end: elapsedSec,
          isSpeech: true,
        });
      }

      const features = analyzeAudioFrame(frame, sampleRate);
      const acoustic = scoreAcoustic(features);
      const timing = scoreTiming(simulatedSegments, elapsedSec);
      const context = scoreContext(actionTypeRef.current);
      const result = fuseScores(acoustic, timing, context);

      onAnalysisResult(result);
      onVisualizationData({
        analyserNode: analyser,
        rmsLevel: rms,
      });

      if (result.tier === 'RED' && onTriggerStepUp) {
        onTriggerStepUp();
      }

      dispatchScoreToServer(result);
    }, 200);
  };

  const handleRunBenchmark = (type: 'legit' | 'clone' | 'noisy') => {
    const ctx = new AudioContext();
    if (type === 'legit') {
      const buf = generateNaturalSpeechBuffer(ctx, 6.0);
      playAudioBufferStream(buf, 'Authentic Natural Caller (Formant Harmonic)');
    } else if (type === 'clone') {
      const buf = generateClonedVoiceBuffer(ctx, 6.0);
      playAudioBufferStream(buf, 'Voice Clone Attack (XTTS-v2 / Neural Vocoder)');
    } else {
      const buf = generateNoisyCallerBuffer(ctx, 6.0);
      playAudioBufferStream(buf, 'Noisy Legitimate Caller (Dadar Station)');
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const arrayBuffer = await file.arrayBuffer();
      const ctx = new AudioContext();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      playAudioBufferStream(audioBuffer, `Uploaded: ${file.name}`);
    } catch {
      alert('Could not decode audio file. Please upload a standard WAV, MP3, or OGG.');
    }
  };

  const handleStop = () => {
    if (processorRef.current) {
      processorRef.current.stop();
      processorRef.current = null;
    }
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.stop();
      } catch {}
      sourceNodeRef.current = null;
    }
    if (benchmarkTimerRef.current) {
      clearInterval(benchmarkTimerRef.current);
      benchmarkTimerRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setActiveBenchmarkName('');
    setStatus('idle');
    setPlaybackProgress(0);
    setPlaybackDuration(0);
    setPlaybackCurrentTime(0);
    onActiveChange(false);
    onVisualizationData({ analyserNode: null, rmsLevel: 0 });
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60)
      .toString()
      .padStart(2, '0');
    const s = Math.floor(secs % 60)
      .toString()
      .padStart(2, '0');
    return `${m}:${s}`;
  };

  const permissionIcon =
    micPermission === 'granted'
      ? '🟢'
      : micPermission === 'denied'
      ? '🔴'
      : '🟡';

  return (
    <div className="rounded-xl border border-[var(--card-border)] bg-[var(--card-bg)] p-4 shadow-sm">
      {/* Mode Selector Tabs */}
      <div className="flex items-center justify-between border-b border-[var(--card-border)] pb-3 mb-3">
        <div className="flex items-center gap-1.5 bg-[#060a14] p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => {
              if (isActive) handleStop();
              setSourceMode('mic');
            }}
            className={`px-3 py-1 rounded text-xs font-semibold transition ${
              sourceMode === 'mic'
                ? 'bg-[var(--cyan)] text-black'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            🎙️ Live Mic Ingress
          </button>
          <button
            onClick={() => {
              if (isActive) handleStop();
              setSourceMode('benchmark');
            }}
            className={`px-3 py-1 rounded text-xs font-semibold transition ${
              sourceMode === 'benchmark'
                ? 'bg-[var(--cyan)] text-black'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            ⚡ Test Benchmarks
          </button>
          <button
            onClick={() => {
              if (isActive) handleStop();
              setSourceMode('file');
            }}
            className={`px-3 py-1 rounded text-xs font-semibold transition ${
              sourceMode === 'file'
                ? 'bg-[var(--cyan)] text-black'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            📁 Upload Audio File
          </button>
        </div>

        {/* Live indicators */}
        <div className="flex items-center gap-3">
          {sourceMode === 'mic' && (
            <div className="flex items-center gap-1.5">
              <span className="text-sm">{permissionIcon}</span>
              <span className="text-[10px] uppercase tracking-wider text-slate-400">
                Mic {micPermission}
              </span>
            </div>
          )}

          {isActive && (
            <div className="flex items-center gap-2">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-green-400 animate-pulse-recording" />
              <span className="font-mono text-base font-bold text-[var(--cyan)]">
                {formatTime(sourceMode === 'mic' ? elapsed : playbackCurrentTime)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Active benchmark label */}
      {isActive && activeBenchmarkName && (
        <div className="mb-3 rounded-lg bg-[var(--cyan)]/10 border border-[var(--cyan)]/30 px-3 py-1.5 text-xs text-[var(--cyan)] flex items-center justify-between">
          <span className="truncate max-w-[280px]">Active Stream: <strong>{activeBenchmarkName}</strong></span>
          <span className="font-mono text-[10px] uppercase">G.711 Telephony Emulation</span>
        </div>
      )}

      {/* Mode Body */}
      {sourceMode === 'mic' ? (
        <div>
          <button
            onClick={isActive ? () => { handleStop(); if (onAudioComplete) onAudioComplete(); } : handleStartMic}
            className={`relative w-full rounded-xl py-3.5 text-center font-semibold transition-all duration-300 ${
              isActive
                ? 'bg-red-500/20 border border-red-500/40 text-red-300 hover:bg-red-500/30'
                : 'bg-[var(--cyan)]/10 border border-[var(--cyan)]/30 text-[var(--cyan)] hover:bg-[var(--cyan)]/20 hover:shadow-lg'
            }`}
          >
            <span className="text-xl mr-2">{isActive ? '⏹' : '🎙️'}</span>
            <span className="text-xs uppercase tracking-widest font-bold">
              {isActive ? 'Stop Telephony Ingress & Finalize Verdict' : 'Start Live Telephony Ingress (Mic)'}
            </span>
          </button>
          <p className="mt-2 text-center text-[10px] text-slate-500">
            Real-time 8kHz G.711 downsampling • 800ms sliding window with 200ms hop • Silero VAD
          </p>
        </div>
      ) : sourceMode === 'benchmark' ? (
        <div className="flex flex-col gap-2">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <button
              onClick={() => handleRunBenchmark('legit')}
              className="rounded-lg border border-green-500/30 bg-green-500/10 hover:bg-green-500/20 p-2.5 text-left transition"
            >
              <div className="text-xs font-bold text-green-300">👤 Genuine Caller</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Natural formants & F0 jitter</div>
              <div className="text-[9px] text-green-400 mt-1 font-mono">Expected: GREEN (&lt;0.30)</div>
            </button>

            <button
              onClick={() => handleRunBenchmark('clone')}
              className="rounded-lg border border-red-500/30 bg-red-500/10 hover:bg-red-500/20 p-2.5 text-left transition"
            >
              <div className="text-xs font-bold text-red-300">🎭 Cloned Voice (XTTS-v2)</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Phase smearing & dead air</div>
              <div className="text-[9px] text-red-400 mt-1 font-mono">Expected: RED (&ge;0.65)</div>
            </button>

            <button
              onClick={() => handleRunBenchmark('noisy')}
              className="rounded-lg border border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/20 p-2.5 text-left transition"
            >
              <div className="text-xs font-bold text-amber-300">🔊 Noisy Station Call</div>
              <div className="text-[10px] text-slate-400 mt-0.5">15dB Dadar noise buffer</div>
              <div className="text-[9px] text-amber-400 mt-1 font-mono">Expected: AMBER / GREEN</div>
            </button>
          </div>

          {/* Progress Bar for Active Benchmark */}
          {isActive && (
            <div className="mt-2 rounded-xl bg-slate-900/60 border border-slate-800 p-3 flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">Analyzing Benchmark Audio</span>
                <span className="font-mono text-[var(--cyan)] font-bold">
                  {formatTime(playbackCurrentTime)} / {formatTime(playbackDuration)} ({playbackProgress}%)
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-[var(--cyan)] to-green-400 h-2 rounded-full transition-all duration-200"
                  style={{ width: `${playbackProgress}%` }}
                />
              </div>
              <button
                onClick={() => {
                  handleStop();
                  if (onAudioComplete) onAudioComplete();
                }}
                className="w-full mt-1 rounded-lg border border-red-500/40 bg-red-500/20 py-1.5 text-xs font-bold text-red-300 hover:bg-red-500/30 transition flex items-center justify-center gap-1.5"
              >
                <span>⏹</span> Stop Benchmark & Finalize Verdict
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-xl p-4 hover:border-[var(--cyan)] transition bg-slate-900/30">
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          {!isActive ? (
            <>
              <span className="text-3xl mb-1">📤</span>
              <p className="text-xs font-semibold text-slate-200">
                Upload Phone Call Audio File (.wav, .mp3, .ogg, .m4a)
              </p>
              <p className="text-[10px] text-slate-400 mt-0.5 mb-3 text-center max-w-sm">
                Plays once through real-time 800ms frames and automatically triggers the ML pipeline to evaluate the final call verdict when complete.
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="rounded-lg bg-[var(--cyan)] px-5 py-2 text-xs font-bold text-black hover:bg-cyan-300 transition shadow-md flex items-center gap-2"
              >
                <span>📁</span> Select Audio File
              </button>
            </>
          ) : (
            <div className="w-full flex flex-col gap-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-semibold truncate max-w-[220px]">
                  ▶️ {activeBenchmarkName || 'Analyzing Audio'}
                </span>
                <span className="font-mono text-[var(--cyan)] font-bold">
                  {formatTime(playbackCurrentTime)} / {formatTime(playbackDuration)} ({playbackProgress}%)
                </span>
              </div>

              {/* Real-time file progress bar */}
              <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-[var(--cyan)] to-green-400 h-2.5 rounded-full transition-all duration-200"
                  style={{ width: `${playbackProgress}%` }}
                />
              </div>

              <div className="flex items-center justify-between mt-1">
                <span className="text-[10px] text-slate-400">
                  Plays to end &bull; Auto-evaluates final overall verdict
                </span>
                <button
                  onClick={() => {
                    handleStop();
                    if (onAudioComplete) onAudioComplete();
                  }}
                  className="rounded-lg bg-red-500/20 border border-red-500/40 text-red-300 hover:bg-red-500/30 px-3 py-1 text-xs font-bold transition flex items-center gap-1"
                >
                  <span>⏹</span> Stop & Finalize Verdict Now
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
