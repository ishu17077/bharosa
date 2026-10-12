// Client-side audio processor that manages:
// - Microphone access via getUserMedia
// - AudioWorklet setup
// - Downsampling to 8kHz (simulating telephony)
// - 800ms sliding window with 200ms hop
// - VAD (Voice Activity Detection) using energy threshold
// - Dispatching audio frames for analysis

import type { SpeechSegment } from './types';

export type AudioProcessorCallback = (data: {
  audioData: Float32Array;
  sampleRate: number;
  timestamp: number;
  visualizationFrame: Float32Array;
  speechSegments: SpeechSegment[];
  isSpeech: boolean;
  rmsLevel: number;
}) => void;

export class AudioProcessor {
  private audioContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private analyserNode: AnalyserNode | null = null;

  // Ring buffer for 800ms sliding window
  private ringBuffer: Float32Array;
  private ringWriteIndex = 0;
  private ringBufferFull = false;

  // VAD state
  private speechSegments: SpeechSegment[] = [];
  private isSpeaking = false;
  private speechStartTime = 0;
  private silenceStartTime = 0;
  private readonly SILENCE_THRESHOLD = 0.015;
  private readonly SPEECH_MIN_DURATION = 100; // ms

  // Hop counter for 200ms hop
  private hopAccumulator = 0;
  private readonly HOP_SIZE_MS = 200;
  private readonly WINDOW_SIZE_MS = 800;
  private readonly TARGET_SAMPLE_RATE = 8000;

  private callback: AudioProcessorCallback | null = null;
  private startTime = 0;
  private isRunning = false;

  // Store the latest visualization frame for callback dispatch
  private latestVisualizationFrame: Float32Array = new Float32Array(0);

  constructor() {
    // Ring buffer sized for 800ms at 8kHz = 6400 samples
    this.ringBuffer = new Float32Array(
      this.TARGET_SAMPLE_RATE * (this.WINDOW_SIZE_MS / 1000)
    );
  }

  async start(callback: AudioProcessorCallback): Promise<void> {
    if (this.isRunning) {
      throw new Error('AudioProcessor is already running');
    }

    this.callback = callback;
    this.startTime = performance.now();
    this.speechSegments = [];
    this.isSpeaking = false;
    this.ringWriteIndex = 0;
    this.ringBufferFull = false;
    this.hopAccumulator = 0;

    try {
      // 1. Create AudioContext
      this.audioContext = new AudioContext();

      // 2. Load AudioWorklet from '/audio-worklet.js'
      await this.audioContext.audioWorklet.addModule('/audio-worklet.js');

      // 3. Get user media (mono mic)
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // 4. Create nodes
      this.sourceNode = this.audioContext.createMediaStreamSource(
        this.mediaStream
      );

      this.analyserNode = this.audioContext.createAnalyser();
      this.analyserNode.fftSize = 2048;
      this.analyserNode.smoothingTimeConstant = 0.8;

      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'bharosa-audio-processor'
      );

      // 5. Connect: mic -> source -> analyser -> worklet
      this.sourceNode.connect(this.analyserNode);
      this.analyserNode.connect(this.workletNode);

      // 6. Set up worklet message handler
      this.workletNode.port.onmessage = (event: MessageEvent) => {
        const { type } = event.data;

        if (type === 'audio-buffer') {
          this.processAudioBuffer(
            event.data.buffer,
            event.data.sampleRate,
            event.data.timestamp
          );
        } else if (type === 'audio-frame') {
          // Store latest frame for visualization
          this.latestVisualizationFrame = event.data.frame;
        }
      };

      this.isRunning = true;
    } catch (error) {
      // Clean up on failure
      this.stop();
      throw error;
    }
  }

  stop(): void {
    this.isRunning = false;

    // Signal the worklet to stop
    if (this.workletNode) {
      this.workletNode.port.postMessage({ command: 'stop' });
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.analyserNode) {
      this.analyserNode.disconnect();
      this.analyserNode = null;
    }

    // Stop all media tracks
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close().catch(() => {
        // Ignore close errors
      });
      this.audioContext = null;
    }

    // Finalize any open speech segment
    if (this.isSpeaking) {
      const elapsed = performance.now() - this.startTime;
      this.speechSegments.push({
        start: this.speechStartTime,
        end: elapsed,
        isSpeech: true,
      });
      this.isSpeaking = false;
    }

    this.callback = null;
  }

  private downsampleTo8kHz(
    buffer: Float32Array,
    originalSampleRate: number
  ): Float32Array {
    if (originalSampleRate === this.TARGET_SAMPLE_RATE) {
      return buffer;
    }

    const ratio = originalSampleRate / this.TARGET_SAMPLE_RATE;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, buffer.length - 1);
      const fraction = srcIndex - srcIndexFloor;

      // Linear interpolation
      result[i] =
        buffer[srcIndexFloor] * (1 - fraction) +
        buffer[srcIndexCeil] * fraction;
    }

    return result;
  }

  private processAudioBuffer(
    buffer: Float32Array,
    originalSampleRate: number,
    timestamp: number
  ): void {
    if (!this.callback) return;

    // 1. Downsample to 8kHz
    const downsampled = this.downsampleTo8kHz(buffer, originalSampleRate);

    // 2. Push into ring buffer
    for (let i = 0; i < downsampled.length; i++) {
      this.ringBuffer[this.ringWriteIndex] = downsampled[i];
      this.ringWriteIndex++;

      if (this.ringWriteIndex >= this.ringBuffer.length) {
        this.ringWriteIndex = 0;
        this.ringBufferFull = true;
      }
    }

    // 3. Run VAD
    const { isSpeech, rmsLevel } = this.detectVAD(downsampled);

    // Track speech segments
    const elapsed = performance.now() - this.startTime;
    if (isSpeech && !this.isSpeaking) {
      this.isSpeaking = true;
      this.speechStartTime = elapsed;
    } else if (!isSpeech && this.isSpeaking) {
      const duration = elapsed - this.speechStartTime;
      if (duration >= this.SPEECH_MIN_DURATION) {
        this.speechSegments.push({
          start: this.speechStartTime,
          end: elapsed,
          isSpeech: true,
        });
      }
      this.isSpeaking = false;
      this.silenceStartTime = elapsed;
    }

    // 4. Every 200ms (hop), extract 800ms window and call the callback
    const bufferDurationMs =
      (downsampled.length / this.TARGET_SAMPLE_RATE) * 1000;
    this.hopAccumulator += bufferDurationMs;

    if (this.hopAccumulator >= this.HOP_SIZE_MS) {
      this.hopAccumulator -= this.HOP_SIZE_MS;

      // Extract the 800ms window from ring buffer
      const windowSize = this.ringBuffer.length;
      const windowData = new Float32Array(windowSize);

      if (this.ringBufferFull) {
        // Ring buffer has wrapped - read from writeIndex to end, then start to writeIndex
        const firstChunkSize = windowSize - this.ringWriteIndex;
        windowData.set(
          this.ringBuffer.subarray(this.ringWriteIndex, windowSize),
          0
        );
        windowData.set(
          this.ringBuffer.subarray(0, this.ringWriteIndex),
          firstChunkSize
        );
      } else {
        // Ring buffer hasn't wrapped yet - just copy what we have
        windowData.set(this.ringBuffer.subarray(0, this.ringWriteIndex));
      }

      this.callback({
        audioData: windowData,
        sampleRate: this.TARGET_SAMPLE_RATE,
        timestamp,
        visualizationFrame: this.latestVisualizationFrame,
        speechSegments: [...this.speechSegments],
        isSpeech,
        rmsLevel,
      });
    }
  }

  private detectVAD(samples: Float32Array): {
    isSpeech: boolean;
    rmsLevel: number;
  } {
    // Calculate RMS energy
    let sumSquares = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSquares += samples[i] * samples[i];
    }
    const rmsLevel = Math.sqrt(sumSquares / samples.length);

    // Simple energy-based VAD
    const isSpeech = rmsLevel > this.SILENCE_THRESHOLD;

    return { isSpeech, rmsLevel };
  }

  getAnalyserNode(): AnalyserNode | null {
    return this.analyserNode;
  }

  getAudioContext(): AudioContext | null {
    return this.audioContext;
  }

  isActive(): boolean {
    return this.isRunning;
  }
}
