// Bharosa Voice Risk Co-Pilot - Digital Signal Processing Library
// Pure client-side audio analysis with manual FFT implementation

import type { AcousticFeatures } from './types';

// ─── Windowing ───────────────────────────────────────────────────────────────

/**
 * Generate a Hann window of given length.
 * w(n) = 0.5 * (1 - cos(2πn / (N-1)))
 */
export function hannWindow(length: number): Float32Array {
  const window = new Float32Array(length);
  const factor = (2 * Math.PI) / (length - 1);
  for (let i = 0; i < length; i++) {
    window[i] = 0.5 * (1 - Math.cos(factor * i));
  }
  return window;
}

/**
 * Apply a window function to samples in-place or to a new array.
 */
export function applyWindow(samples: Float32Array, window: Float32Array): Float32Array {
  const N = Math.min(samples.length, window.length);
  const result = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    result[i] = samples[i] * window[i];
  }
  return result;
}

// ─── FFT (Radix-2 Cooley-Tukey) ─────────────────────────────────────────────

/**
 * Next power of 2 greater than or equal to n.
 */
function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Bit-reversal permutation for radix-2 FFT.
 */
function bitReverse(arr: Float32Array, n: number): void {
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      // Swap real parts
      const tmpR = arr[2 * i];
      arr[2 * i] = arr[2 * j];
      arr[2 * j] = tmpR;
      // Swap imaginary parts
      const tmpI = arr[2 * i + 1];
      arr[2 * i + 1] = arr[2 * j + 1];
      arr[2 * j + 1] = tmpI;
    }
  }
}

/**
 * In-place radix-2 Cooley-Tukey FFT.
 * Input: interleaved [re0, im0, re1, im1, ...] of length 2*N
 * N must be a power of 2.
 */
export function computeFFT(realImag: Float32Array, N: number): void {
  bitReverse(realImag, N);

  for (let size = 2; size <= N; size <<= 1) {
    const halfSize = size >> 1;
    const angle = (-2 * Math.PI) / size;

    // Precompute twiddle factors for this stage
    const wR = Math.cos(angle);
    const wI = Math.sin(angle);

    for (let start = 0; start < N; start += size) {
      let curR = 1.0;
      let curI = 0.0;

      for (let k = 0; k < halfSize; k++) {
        const evenIdx = 2 * (start + k);
        const oddIdx = 2 * (start + k + halfSize);

        const eR = realImag[evenIdx];
        const eI = realImag[evenIdx + 1];

        const tR = curR * realImag[oddIdx] - curI * realImag[oddIdx + 1];
        const tI = curR * realImag[oddIdx + 1] + curI * realImag[oddIdx];

        realImag[evenIdx] = eR + tR;
        realImag[evenIdx + 1] = eI + tI;
        realImag[oddIdx] = eR - tR;
        realImag[oddIdx + 1] = eI - tI;

        // Rotate twiddle factor
        const newR = curR * wR - curI * wI;
        const newI = curR * wI + curI * wR;
        curR = newR;
        curI = newI;
      }
    }
  }
}

/**
 * Compute the magnitude spectrum from FFT output.
 * Returns only the first N/2 + 1 bins (positive frequencies).
 */
export function computeMagnitudeSpectrum(realImag: Float32Array, N: number): Float32Array {
  const numBins = (N >> 1) + 1;
  const magnitudes = new Float32Array(numBins);
  for (let i = 0; i < numBins; i++) {
    const re = realImag[2 * i];
    const im = realImag[2 * i + 1];
    magnitudes[i] = Math.sqrt(re * re + im * im);
  }
  return magnitudes;
}

/**
 * Compute the power spectrum from FFT output.
 * Returns only the first N/2 + 1 bins (positive frequencies).
 */
export function computePowerSpectrum(realImag: Float32Array, N: number): Float32Array {
  const numBins = (N >> 1) + 1;
  const power = new Float32Array(numBins);
  for (let i = 0; i < numBins; i++) {
    const re = realImag[2 * i];
    const im = realImag[2 * i + 1];
    power[i] = re * re + im * im;
  }
  return power;
}

// ─── Spectral Features ──────────────────────────────────────────────────────

/**
 * Spectral flatness: geometric_mean / arithmetic_mean of power spectrum.
 * A flat (noise-like) spectrum yields ~1; a tonal spectrum yields ~0.
 * Synthetic voices often have unnaturally smooth spectra (higher flatness).
 */
function computeSpectralFlatness(powerSpectrum: Float32Array): number {
  const N = powerSpectrum.length;
  if (N === 0) return 0;

  // Use log-domain computation for geometric mean to avoid overflow/underflow
  let logSum = 0;
  let arithmeticSum = 0;
  let validCount = 0;

  for (let i = 0; i < N; i++) {
    const val = powerSpectrum[i];
    if (val > 1e-20) {
      logSum += Math.log(val);
      validCount++;
    } else {
      logSum += Math.log(1e-20);
      validCount++;
    }
    arithmeticSum += val;
  }

  if (validCount === 0 || arithmeticSum < 1e-20) return 0;

  const geometricMean = Math.exp(logSum / validCount);
  const arithmeticMean = arithmeticSum / validCount;

  return Math.min(1, geometricMean / arithmeticMean);
}

/**
 * Spectral centroid: weighted mean of frequencies by magnitude.
 * Indicates the "brightness" of the signal.
 */
function computeSpectralCentroid(magnitudeSpectrum: Float32Array, sampleRate: number, fftSize: number): number {
  const N = magnitudeSpectrum.length;
  let numerator = 0;
  let denominator = 0;
  const freqResolution = sampleRate / fftSize;

  for (let i = 0; i < N; i++) {
    const freq = i * freqResolution;
    numerator += freq * magnitudeSpectrum[i];
    denominator += magnitudeSpectrum[i];
  }

  return denominator > 1e-10 ? numerator / denominator : 0;
}

/**
 * Spectral rolloff: frequency below which a specified percentage of total spectral energy lies.
 * Default threshold is 85%.
 */
function computeSpectralRolloff(powerSpectrum: Float32Array, sampleRate: number, fftSize: number, threshold: number = 0.85): number {
  const N = powerSpectrum.length;
  let totalEnergy = 0;

  for (let i = 0; i < N; i++) {
    totalEnergy += powerSpectrum[i];
  }

  const target = totalEnergy * threshold;
  let cumulativeEnergy = 0;
  const freqResolution = sampleRate / fftSize;

  for (let i = 0; i < N; i++) {
    cumulativeEnergy += powerSpectrum[i];
    if (cumulativeEnergy >= target) {
      return i * freqResolution;
    }
  }

  return (N - 1) * freqResolution;
}

/**
 * Spectral bandwidth: weighted standard deviation of frequencies around the centroid.
 */
function computeSpectralBandwidth(magnitudeSpectrum: Float32Array, sampleRate: number, fftSize: number, centroid: number): number {
  const N = magnitudeSpectrum.length;
  let numerator = 0;
  let denominator = 0;
  const freqResolution = sampleRate / fftSize;

  for (let i = 0; i < N; i++) {
    const freq = i * freqResolution;
    const diff = freq - centroid;
    numerator += magnitudeSpectrum[i] * diff * diff;
    denominator += magnitudeSpectrum[i];
  }

  return denominator > 1e-10 ? Math.sqrt(numerator / denominator) : 0;
}

// ─── Time-Domain Features ───────────────────────────────────────────────────

/**
 * Zero crossing rate: proportion of signal sign changes per sample.
 */
function computeZeroCrossingRate(samples: Float32Array): number {
  const N = samples.length;
  if (N < 2) return 0;

  let crossings = 0;
  for (let i = 1; i < N; i++) {
    if ((samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0)) {
      crossings++;
    }
  }

  return crossings / (N - 1);
}

/**
 * Compute variance of zero crossing rates across sub-frames.
 * Low variance indicates unnaturally regular signal (synthetic).
 */
function computeZCRVariance(samples: Float32Array, frameSize: number = 256): number {
  const N = samples.length;
  const numFrames = Math.floor(N / frameSize);
  if (numFrames < 2) return 0;

  const zcrs: number[] = [];
  for (let f = 0; f < numFrames; f++) {
    const start = f * frameSize;
    const frame = samples.subarray(start, start + frameSize);
    zcrs.push(computeZeroCrossingRate(frame));
  }

  const mean = zcrs.reduce((a, b) => a + b, 0) / zcrs.length;
  const variance = zcrs.reduce((sum, z) => sum + (z - mean) * (z - mean), 0) / zcrs.length;
  return variance;
}

/**
 * RMS energy of the signal.
 */
function computeRMSEnergy(samples: Float32Array): number {
  const N = samples.length;
  if (N === 0) return 0;

  let sumSquares = 0;
  for (let i = 0; i < N; i++) {
    sumSquares += samples[i] * samples[i];
  }

  return Math.sqrt(sumSquares / N);
}

// ─── Pitch & HNR ────────────────────────────────────────────────────────────

/**
 * Compute autocorrelation of a signal.
 * Returns autocorrelation values for lags 0..maxLag.
 */
function computeAutocorrelation(samples: Float32Array, maxLag: number): Float32Array {
  const N = samples.length;
  const result = new Float32Array(maxLag + 1);

  for (let lag = 0; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < N - lag; i++) {
      sum += samples[i] * samples[i + lag];
    }
    result[lag] = sum;
  }

  return result;
}

/**
 * Autocorrelation-based pitch detection.
 * Returns detected pitch in Hz, or 0 if no clear pitch.
 * Search range: 80 Hz to 500 Hz (human voice range).
 */
function detectPitch(samples: Float32Array, sampleRate: number): number {
  const minPeriod = Math.floor(sampleRate / 500); // 500 Hz upper bound
  const maxPeriod = Math.floor(sampleRate / 80);  // 80 Hz lower bound
  const maxLag = Math.min(maxPeriod, samples.length - 1);

  if (maxLag <= minPeriod) return 0;

  const autocorr = computeAutocorrelation(samples, maxLag);

  // Normalize by zero-lag (energy)
  const r0 = autocorr[0];
  if (r0 < 1e-10) return 0;

  // Find the peak in the autocorrelation within the valid range
  let bestLag = 0;
  let bestVal = -Infinity;

  for (let lag = minPeriod; lag <= maxLag; lag++) {
    const normalized = autocorr[lag] / r0;
    if (normalized > bestVal) {
      bestVal = normalized;
      bestLag = lag;
    }
  }

  // Require a minimum correlation strength for a valid pitch
  if (bestVal < 0.3 || bestLag === 0) return 0;

  // Parabolic interpolation for sub-sample accuracy
  if (bestLag > minPeriod && bestLag < maxLag) {
    const a = autocorr[bestLag - 1];
    const b = autocorr[bestLag];
    const c = autocorr[bestLag + 1];
    const denominator = 2 * (2 * b - a - c);
    if (Math.abs(denominator) > 1e-10) {
      const delta = (a - c) / denominator;
      const refinedLag = bestLag + delta;
      return sampleRate / refinedLag;
    }
  }

  return sampleRate / bestLag;
}

/**
 * Compute pitch contour variance across sub-frames.
 * Low variance indicates monotonous/robotic pitch (synthetic).
 */
function computePitchContourVariance(samples: Float32Array, sampleRate: number, frameSize: number = 1024): number {
  const N = samples.length;
  const hopSize = frameSize >> 1;
  const numFrames = Math.max(1, Math.floor((N - frameSize) / hopSize) + 1);

  const pitches: number[] = [];
  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    const end = Math.min(start + frameSize, N);
    const frame = samples.subarray(start, end);
    const pitch = detectPitch(frame, sampleRate);
    if (pitch > 0) {
      pitches.push(pitch);
    }
  }

  if (pitches.length < 2) return 0;

  const mean = pitches.reduce((a, b) => a + b, 0) / pitches.length;
  const variance = pitches.reduce((sum, p) => sum + (p - mean) * (p - mean), 0) / pitches.length;

  // Normalize by mean squared to get coefficient of variation squared
  return mean > 0 ? variance / (mean * mean) : 0;
}

/**
 * Estimate Harmonic-to-Noise Ratio (HNR) using autocorrelation method.
 * HNR = 10 * log10(r_peak / (r_0 - r_peak))
 * where r_peak is the autocorrelation at the pitch period.
 * Low HNR indicates noisy/synthetic voice with phase artifacts.
 */
function computeHNR(samples: Float32Array, sampleRate: number): number {
  const minPeriod = Math.floor(sampleRate / 500);
  const maxPeriod = Math.floor(sampleRate / 80);
  const maxLag = Math.min(maxPeriod, samples.length - 1);

  if (maxLag <= minPeriod) return 0;

  const autocorr = computeAutocorrelation(samples, maxLag);
  const r0 = autocorr[0];

  if (r0 < 1e-10) return 0;

  // Find peak in pitch range
  let rPeak = -Infinity;
  for (let lag = minPeriod; lag <= maxLag; lag++) {
    if (autocorr[lag] > rPeak) {
      rPeak = autocorr[lag];
    }
  }

  // HNR in dB
  const noise = r0 - rPeak;
  if (noise <= 1e-10) return 40; // Very high HNR (nearly pure tone)
  if (rPeak <= 0) return 0;      // No harmonic content

  return 10 * Math.log10(rPeak / noise);
}

/**
 * Compute spectral centroid variance across sub-frames.
 * Low variance indicates unnaturally stable spectral content (synthetic).
 */
function computeSpectralCentroidVariance(samples: Float32Array, sampleRate: number, frameSize: number = 1024): number {
  const N = samples.length;
  const fftSize = nextPowerOf2(frameSize);
  const hopSize = frameSize >> 1;
  const numFrames = Math.max(1, Math.floor((N - frameSize) / hopSize) + 1);
  const window = hannWindow(frameSize);

  const centroids: number[] = [];

  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    const end = Math.min(start + frameSize, N);
    const frame = samples.subarray(start, end);

    // Zero-pad to FFT size
    const padded = new Float32Array(fftSize);
    const windowed = applyWindow(frame, window);
    padded.set(windowed);

    // Prepare interleaved complex array
    const complex = new Float32Array(fftSize * 2);
    for (let i = 0; i < fftSize; i++) {
      complex[2 * i] = padded[i];
      complex[2 * i + 1] = 0;
    }

    computeFFT(complex, fftSize);
    const mag = computeMagnitudeSpectrum(complex, fftSize);
    centroids.push(computeSpectralCentroid(mag, sampleRate, fftSize));
  }

  if (centroids.length < 2) return 0;

  const mean = centroids.reduce((a, b) => a + b, 0) / centroids.length;
  const variance = centroids.reduce((sum, c) => sum + (c - mean) * (c - mean), 0) / centroids.length;

  // Normalize by mean squared
  return mean > 0 ? variance / (mean * mean) : 0;
}

// ─── Main Analysis Function ─────────────────────────────────────────────────

/**
 * Analyze a single audio frame and extract all acoustic features.
 * This is the primary entry point for the DSP pipeline.
 *
 * @param samples - Float32Array of PCM audio samples (mono, normalized to [-1, 1])
 * @param sampleRate - Sample rate in Hz (e.g. 16000, 44100, 48000)
 * @returns AcousticFeatures object with all computed features
 */
export function analyzeAudioFrame(samples: Float32Array, sampleRate: number): AcousticFeatures {
  const N = samples.length;

  if (N === 0) {
    return {
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
    };
  }

  // ── Time-domain features ──
  const zeroCrossingRate = computeZeroCrossingRate(samples);
  const zeroCrossingRateVariance = computeZCRVariance(samples);
  const rmsEnergy = computeRMSEnergy(samples);

  // ── FFT for spectral features ──
  const fftSize = nextPowerOf2(N);
  const window = hannWindow(N);
  const windowed = applyWindow(samples, window);

  // Zero-pad windowed signal to FFT size
  const padded = new Float32Array(fftSize);
  padded.set(windowed);

  // Prepare interleaved complex array [re0, im0, re1, im1, ...]
  const complex = new Float32Array(fftSize * 2);
  for (let i = 0; i < fftSize; i++) {
    complex[2 * i] = padded[i];
    complex[2 * i + 1] = 0;
  }

  computeFFT(complex, fftSize);

  const magnitudeSpectrum = computeMagnitudeSpectrum(complex, fftSize);
  const powerSpectrum = computePowerSpectrum(complex, fftSize);

  // ── Spectral features ──
  const spectralFlatness = computeSpectralFlatness(powerSpectrum);
  const spectralCentroid = computeSpectralCentroid(magnitudeSpectrum, sampleRate, fftSize);
  const spectralRolloff = computeSpectralRolloff(powerSpectrum, sampleRate, fftSize);
  const spectralBandwidth = computeSpectralBandwidth(magnitudeSpectrum, sampleRate, fftSize, spectralCentroid);

  // ── Cross-frame variance features ──
  const spectralCentroidVariance = computeSpectralCentroidVariance(samples, sampleRate);
  const pitchContourVariance = computePitchContourVariance(samples, sampleRate);

  // ── Harmonic-to-Noise Ratio ──
  const harmonicToNoiseRatio = computeHNR(samples, sampleRate);

  return {
    spectralFlatness,
    zeroCrossingRate,
    zeroCrossingRateVariance,
    spectralCentroid,
    spectralCentroidVariance,
    spectralRolloff,
    harmonicToNoiseRatio,
    pitchContourVariance,
    rmsEnergy,
    spectralBandwidth,
  };
}
