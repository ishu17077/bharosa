// Bharosa Telephony Benchmark Audio Synthesis
// Generates realistic PCM audio buffers for reproducible testing without requiring external audio files

/**
 * Generate a synthetic speech buffer representing a natural human caller:
 * - Natural vocal tract formants (F1, F2, F3)
 * - F0 pitch contour with natural human jitter, prosody drift, micro-tremors
 * - Natural phonemic zero-crossing rate variation
 * - Soft ambient room acoustics (25dB SNR)
 */
export function generateNaturalSpeechBuffer(
  audioCtx: AudioContext,
  durationSec: number = 4.0
): AudioBuffer {
  const sampleRate = audioCtx.sampleRate;
  const numSamples = Math.floor(sampleRate * durationSec);
  const buffer = audioCtx.createBuffer(1, numSamples, sampleRate);
  const data = buffer.getChannelData(0);

  let phase = 0;
  let f0 = 135; // Base pitch (Hz)

  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;

    // Conversational pauses
    const isPause = (t > 1.2 && t < 1.7) || (t > 2.8 && t < 3.2);

    if (isPause) {
      // Natural ambient room noise during pause (not dead digital silence)
      data[i] = (Math.random() * 2 - 1) * 0.008;
      continue;
    }

    // Natural prosodic intonation: question rise and phrase declination
    const intonation = Math.sin(t * 3.5) * 18 + Math.sin(t * 7.2) * 6;
    // Micro-jitter
    const jitter = (Math.random() - 0.5) * 4;
    const currentF0 = f0 + intonation + jitter;

    phase += (2 * Math.PI * currentF0) / sampleRate;
    if (phase > 2 * Math.PI) phase -= 2 * Math.PI;

    // Harmonic source (glottal pulse pulse train with rich harmonics)
    let glottal = 0;
    for (let h = 1; h <= 12; h++) {
      const harmAmp = 1 / Math.pow(h, 1.2);
      glottal += Math.sin(h * phase) * harmAmp;
    }

    // Formant filtering approximation (F1: 650Hz, F2: 1200Hz, F3: 2500Hz)
    const formant1 = Math.sin(phase * (650 / currentF0)) * 0.45;
    const formant2 = Math.sin(phase * (1200 / currentF0)) * 0.35;
    const formant3 = Math.sin(phase * (2500 / currentF0)) * 0.2;

    const speech = glottal * 0.3 + formant1 + formant2 + formant3;

    // Envelope (syllabic modulation ~ 4-5 Hz)
    const envelope = Math.max(0, Math.sin(t * 2 * Math.PI * 4.2)) * 0.6 + 0.1;

    // Room acoustics / background dither
    const dither = (Math.random() * 2 - 1) * 0.012;

    data[i] = speech * envelope * 0.4 + dither;
  }

  return buffer;
}

/**
 * Generate a synthetic speech buffer representing a voice-cloned attack (XTTS-v2 / HiFi-GAN):
 * - Unnaturally flat spectral envelope (high spectral flatness)
 * - Rigid, overly-stable fundamental frequency (low pitch variance)
 * - Phase smearing in high frequencies from neural vocoder synthesis
 * - 1.4s generative latency gap with absolute digital dead air (no room reflections)
 */
export function generateClonedVoiceBuffer(
  audioCtx: AudioContext,
  durationSec: number = 4.5
): AudioBuffer {
  const sampleRate = audioCtx.sampleRate;
  const numSamples = Math.floor(sampleRate * durationSec);
  const buffer = audioCtx.createBuffer(1, numSamples, sampleRate);
  const data = buffer.getChannelData(0);

  let phase = 0;
  const rigidF0 = 168.0; // Robotic, locked pitch without natural vibrato/jitter

  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;

    // 1.4s generative lag at beginning (synthesis engine latency) with dead silence
    if (t < 1.4) {
      // Unnatural absolute dead air: exactly 0.000 (no room acoustics)
      data[i] = 0.0;
      continue;
    }

    // Overly uniform pitch (vocoder artifact)
    phase += (2 * Math.PI * rigidF0) / sampleRate;
    if (phase > 2 * Math.PI) phase -= 2 * Math.PI;

    // Vocoder phase smearing: phase cancellation in upper harmonics
    let vocoderAudio = 0;
    for (let h = 1; h <= 16; h++) {
      // Artificially smooth harmonic roll-off creates high spectral flatness
      const harmAmp = 0.75 / Math.pow(h, 0.45);
      // Phase discontinuity artifact: randomize phase slightly in higher harmonics
      const phaseOffset = h > 4 ? Math.sin(h * 13.7) * 1.8 : 0;
      vocoderAudio += Math.sin(h * phase + phaseOffset) * harmAmp;
    }

    // Unnatural envelope with sharp, mechanical edges
    const tActive = t - 1.4;
    const mechanicalEnvelope = (Math.sin(tActive * 2 * Math.PI * 3.8) > 0 ? 0.7 : 0.05);

    // High frequency buzz (HiFi-GAN vocoder artifact)
    const buzz = Math.sin(phase * 18.3) * 0.15;

    data[i] = (vocoderAudio * mechanicalEnvelope * 0.35) + buzz * 0.1;
  }

  return buffer;
}

/**
 * Generate a noisy legitimate caller buffer (e.g. Dadar railway station background):
 * - Speech mixed with 15dB SNR train / crowd rumbling
 * - Preserves natural glottal dynamics and natural phase coherence
 */
export function generateNoisyCallerBuffer(
  audioCtx: AudioContext,
  durationSec: number = 4.0
): AudioBuffer {
  const naturalBuffer = generateNaturalSpeechBuffer(audioCtx, durationSec);
  const data = naturalBuffer.getChannelData(0);

  // Layer in pink/brown noise simulating urban street rumble
  let b0 = 0, b1 = 0, b2 = 0;
  for (let i = 0; i < data.length; i++) {
    const white = Math.random() * 2 - 1;
    b0 = 0.99 * b0 + white * 0.05;
    b1 = 0.95 * b1 + white * 0.1;
    b2 = 0.85 * b2 + white * 0.2;
    const rumble = (b0 + b1 + b2) * 0.07;
    data[i] = data[i] * 0.85 + rumble;
  }

  return naturalBuffer;
}

